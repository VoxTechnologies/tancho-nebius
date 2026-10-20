"""Lossless flight-field adapter from compiled plans to tancho-viewpoints-1.0."""
from .contracts import fields, require
from .mission_context import canonical_sha256
from .px4_mission import FRAME, SCHEMA


def adapt_to_viewpoints_v1(compiled, context):
    fields(compiled, 'schema_version context_id mission_id program_revision envelope_revision '
                     'calibration_id coordinate_frame viewpoints compiled_revision')
    require(compiled['schema_version'] in ('tancho-compiled-viewplan-1.0',
                                           'tancho-compiled-dynamic-search-1.0'),
            'Unknown compiled plan')
    require(compiled['context_id'] == context['context_id']
            and compiled['mission_id'] == context['mission_id'], 'Compiled plan is bound elsewhere')
    require(compiled['envelope_revision'] == context['envelope_revision'], 'Envelope revision changed')
    require(compiled['calibration_id'] == context['calibration_id'], 'Calibration changed')
    require(compiled['coordinate_frame'] == FRAME, 'Unknown compiled coordinate frame')
    rows = []
    for index, row in enumerate(compiled['viewpoints']):
        fields(row, 'sequence observation_sequence intent_id target_id position_m target_position_m '
                    'dwell_ms speed_mps sensor stop_condition')
        require(row['sequence'] == index, 'Compiled viewpoint order changed')
        rows.append({'sequence': index, 'position_m': list(row['position_m']),
                     'target_position_m': list(row['target_position_m']),
                     'dwell_ms': row['dwell_ms'], 'speed_mps': row['speed_mps'],
                     'target_id': row['target_id'], 'evidence_ids': list(context['evidence_ids'])})
    return {'schema_version': SCHEMA, 'observation_id': context['mission_id'],
            'goal_revision': compiled['program_revision'],
            'envelope_revision': compiled['envelope_revision'],
            'calibration_id': compiled['calibration_id'], 'coordinate_frame': FRAME,
            'viewpoints': rows}


def adaptation_record(compiled, viewpoints):
    """Hashes retained beside an adapter result in the audit record."""
    return {'schema_version': 'tancho-viewpoint-adaptation-record-1.0',
            'compiled_revision': compiled['compiled_revision'],
            'compiled_sha256': canonical_sha256(compiled),
            'viewpoints_sha256': canonical_sha256(viewpoints)}
