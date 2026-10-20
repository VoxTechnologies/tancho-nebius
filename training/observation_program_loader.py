"""Native Cosmos data adapter for a verified Observation Program SFT pack."""
import copy
import hashlib
import io
import json
from pathlib import Path

from tancho.mission_context import canonical_sha256

SCHEMA = 'tancho-observation-program-sft-pack-1.0'
DETERMINISTIC_AUTHORITIES = {
    'deterministic_scene_ground_truth': 'synthetic',
    'deterministic_legacy_migration': 'real',
}


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _local(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Invalid Observation Program pack path')
    return path


def verify_pack(root, *, purpose):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema_version') != SCHEMA:
        raise ValueError('Wrong Observation Program pack schema')
    recorded = manifest.get('manifest_sha256')
    body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if recorded != canonical_sha256(body):
        raise ValueError('Observation Program manifest changed')
    actual = {str(path.relative_to(root)) for path in root.rglob('*')
              if path.is_file() and path.name != '.DS_Store'}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Observation Program pack inventory differs')
    for name, digest in manifest['files'].items():
        if _digest(_local(root, name)) != digest:
            raise ValueError('Observation Program payload changed: ' + name)
    for row in manifest['samples']:
        provenance = row.get('provenance', {})
        authority = provenance.get('label_authority')
        if authority not in DETERMINISTIC_AUTHORITIES:
            continue
        proof_sha = provenance.get('label_authority_sha256')
        proof = json.loads(_local(root, f'proofs/{proof_sha}.json').read_text())
        if (canonical_sha256(proof) != proof_sha
                or proof.get('split') != row['split']
                or proof.get('profile') != row['mission_profile']
                or proof.get('role') != row['role']
                or (proof.get('source_media_sha256') is not None
                    and proof.get('source_media_sha256') != row['source_media_sha256'])
                or proof.get('expected_target') != json.loads(row['target'])):
            raise ValueError('Deterministic label proof does not bind the training label')
        if authority == 'deterministic_legacy_migration':
            from training.observation_program_legacy_authority import validate_legacy_proof
            validate_legacy_proof(proof)
    if purpose not in ('training', 'compatibility', 'evaluation'):
        raise ValueError('Unknown Observation Program pack purpose')
    if purpose in ('training', 'compatibility') and not manifest['training_samples']:
        raise ValueError('Observation Program pack has no training samples')
    def trusted_label(row):
        provenance = row.get('provenance', {})
        return (provenance.get('human_reviewed') is True
                or (provenance.get('source_kind') == DETERMINISTIC_AUTHORITIES.get(
                        provenance.get('label_authority'))
                    and provenance.get('human_reviewed') is False
                    and isinstance(provenance.get('label_authority_sha256'), str)
                    and len(provenance['label_authority_sha256']) == 64))
    if purpose == 'training' and any(not trusted_label(row) for row in manifest['samples']
                                     if row['split'] == 'train'):
        raise ValueError('Final training pack contains an unreviewed label')
    split_by_dimension = {}
    for row in manifest['samples']:
        if row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown Observation Program split')
        for key, value in ({'leakage_group_id': row['leakage_group_id']}
                           | row['leakage_dimensions']).items():
            identity = key + ':' + value
            if identity in split_by_dimension and split_by_dimension[identity] != row['split']:
                raise ValueError('Observation Program split leakage')
            split_by_dimension[identity] = row['split']
    return manifest


class ObservationProgramDataset:
    def __init__(self, root, split, *, purpose='training'):
        self.root = Path(root).resolve()
        self.manifest = verify_pack(self.root, purpose=purpose)
        self.rows = [row for row in self.manifest['samples'] if row['split'] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        frames, proof = [], []
        for evidence_id, name in zip(row['frames_sha256'], row['frames'], strict=True):
            data = _local(self.root, name).read_bytes()
            if hashlib.sha256(data).hexdigest() != row['frames_sha256'][evidence_id]:
                raise ValueError('Training frame changed')
            with Image.open(io.BytesIO(data)) as source:
                if source.format != 'PNG' or source.mode != 'RGB' or getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('Training frame is not canonical RGB PNG')
                source.load(); image = source.copy()
            frames.append(image)
            proof.append({'id': evidence_id, 'sha256': row['frames_sha256'][evidence_id],
                          'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                          'size': list(image.size)})
        if len({image.size for image in frames}) != 1:
            raise ValueError('Training frame dimensions differ')
        conversation = json.loads(_local(self.root, row['conversation']).read_text())['conversations']
        if conversation[0]['content'][1]['text'] != row['prompt'] \
                or conversation[1]['content'][0]['text'] != row['target']:
            raise ValueError('Training conversation changed')
        messages = copy.deepcopy(conversation)
        messages[0]['content'][0] = {'type': 'video', 'video': frames, 'fps': .5}
        return {'texts': messages, 'media': {}, '__key__': row['sample_id'],
                'tancho_role': row['role'], 'tancho_profile': row['mission_profile'],
                'canonical_frames': {'frames': proof, 'fps': .5,
                                     'resize_before_processor': False, 'video_decode': False}}


def program_processor_class():
    from cosmos_framework.configs.base.reasoner.experiment.videophy2_dataflow_roles import VideoPhy2Processor

    class ProgramProcessor(VideoPhy2Processor):
        def process(self, item):
            result = super().process(item)
            result['tancho_sample_id'] = item['__key__']
            result['tancho_role'] = item['tancho_role']
            result['tancho_profile'] = item['tancho_profile']
            return result

    return ProgramProcessor


def program_collator_class():
    from cosmos_framework.configs.base.reasoner.experiment.dataflow_roles import VLMCollator

    class ProgramCollator(VLMCollator):
        def collate(self, samples):
            if len(samples) != 1:
                raise ValueError('Observation Program collator requires batch size 1')
            batch = super().collate(samples)
            length = samples[0]['input_ids'].numel()
            for key in ('input_ids', 'labels', 'token_mask', 'attention_mask'):
                batch[key] = batch[key][..., :length]
            return batch

    return ProgramCollator


def configure(config, bundle, *, purpose='training'):
    from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
    from cosmos_framework.data.generator.dataflow import MapDistributor
    from cosmos_framework.utils.lazy_config import LazyCall as L

    ProgramProcessor = program_processor_class()
    ProgramCollator = program_collator_class()

    for key, split in (('dataloader_train', 'train'), ('dataloader_val', 'validation')):
        loader = getattr(config, key)
        loader.distributor = L(MapDistributor)(
            dataset=L(ObservationProgramDataset)(root=str(bundle), split=split, purpose=purpose),
            shuffle=True, seed=42, name='tancho_program_' + split)
        loader.processor['_target_'] = ProgramProcessor
        loader.collator = L(ProgramCollator)()
        loader.num_workers = 0; loader.persistent_workers = False; loader.prefetch_factor = None
        loader.batcher.pool_size = 1; loader.batcher.max_batch_size = 1
    config.trainer.callbacks.dataloader_state = L(CosmosDataLoaderStateCallback)(
        name='tancho_program_train')
