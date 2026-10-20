"""Audited state machine for the Reasoner → Generator → flight boundary."""
import hashlib

from .contracts import ContractError, fields, require
from .mission_context import canonical_sha256, validate_context_v1
from .observation_compiler import compile_program
from .observation_intent import parse_intent_output, validate_and_bind_intents
from .observation_program import parse_program_output, validate_observation_program
from .observation_result import verify_observation_result

TERMINAL = frozenset({'COMPLETE', 'ABORTED', 'DETECTOR_HANDOFF'})


class ObservationProgramRuntime:
    def __init__(self, context, evidence, *, now_ms):
        validate_context_v1(context, now_ms=now_ms, evidence=evidence)
        self.context = context
        self.state = 'CAPTURED'
        self.cycles = 0
        self.intents = self.program = self.compiled = self.result = None
        self.audit = []
        self._event('capture', {'context_id': context['context_id'],
                                'input_sha256': context['input_sha256']})
        self.state = 'CONTEXT_VALIDATED'
        self._event('context_validated', {'context_sha256': canonical_sha256(context)})

    def _event(self, kind, artifact):
        previous = self.audit[-1]['event_sha256'] if self.audit else None
        body = {'sequence': len(self.audit), 'kind': kind, 'state': self.state,
                'previous_event_sha256': previous, 'artifact': artifact}
        self.audit.append({**body, 'event_sha256': canonical_sha256(body)})

    def _at(self, state):
        require(self.state == state, f'Runtime state must be {state}')

    def record_model_receipt(self, receipt):
        require(self.state in ('INTENTS_VERIFIED', 'PROGRAM_VERIFIED'),
                'Model receipt is not adjacent to a verified output')
        expected_role = 'reasoner' if self.state == 'INTENTS_VERIFIED' else 'generator'
        require(receipt.get('role') == expected_role, 'Model receipt role differs from runtime state')
        self._event('model_receipt', dict(receipt))

    def reason(self, raw_text, *, now_ms):
        self._at('CONTEXT_VALIDATED')
        self.state = 'INTENTS_PENDING'
        self._event('reasoner_raw', {'raw_sha256': hashlib.sha256(raw_text.encode()).hexdigest()})
        self.intents = validate_and_bind_intents(parse_intent_output(raw_text), self.context,
                                                 now_ms=now_ms, raw_text=raw_text)
        self.state = 'INTENTS_VERIFIED'
        self._event('intents_verified', {'intent_revision': self.intents['intent_revision'],
                                         'outcome': self.intents['outcome'],
                                         'decision_source': 'tancho_model'})
        return self.intents

    def generate(self, raw_text, *, now_ms):
        self._at('INTENTS_VERIFIED')
        require(self.intents['outcome'] == 'inspect', 'Intent outcome does not request observation')
        self.state = 'PROGRAM_PENDING'
        self._event('generator_raw', {'raw_sha256': hashlib.sha256(raw_text.encode()).hexdigest()})
        self.program = validate_observation_program(parse_program_output(raw_text), self.context,
                                                    self.intents, now_ms=now_ms, raw_text=raw_text)
        self.state = 'PROGRAM_VERIFIED'
        self._event('program_verified', {'program_revision': self.program['program_revision'],
                                         'decision_source': 'tancho_model'})
        return self.program

    def compile(self, scene, vehicle, *, now_ms):
        self._at('PROGRAM_VERIFIED')
        self.compiled = compile_program(self.program, self.context, scene, vehicle, now_ms=now_ms)
        self.state = 'COMPILED'
        self._event('compiled', {'compiled_revision': self.compiled['compiled_revision']})
        return self.compiled

    def approve_safety(self, report):
        self._at('COMPILED')
        fields(report, 'schema_version compiled_revision passed report_sha256')
        require(report['schema_version'] == 'tancho-rfl-report-1.0', 'Unknown RFL report')
        require(report['compiled_revision'] == self.compiled['compiled_revision'], 'RFL checked another plan')
        require(report['passed'] is True, 'RFL rejected the compiled plan')
        require(isinstance(report['report_sha256'], str)
                and len(report['report_sha256']) == 64
                and all(c in '0123456789abcdef' for c in report['report_sha256']),
                'Invalid RFL report hash')
        self.state = 'SAFETY_VALIDATED'
        self._event('safety_validated', {'compiled_revision': report['compiled_revision'],
                                         'rfl_report_sha256': report['report_sha256']})

    def authorize_execution(self):
        self._at('SAFETY_VALIDATED')
        self.state = 'EXECUTING'
        body = {'context_id': self.context['context_id'],
                'compiled_revision': self.compiled['compiled_revision'],
                'audit_head': self.audit[-1]['event_sha256']}
        authorization = {'schema_version': 'tancho-flight-authorization-1.0', **body,
                         'authorization_sha256': canonical_sha256(body)}
        self._event('execution_authorized', authorization)
        return authorization

    def record_px4_prepared(self, mission_sha256):
        self._at('EXECUTING')
        require(isinstance(mission_sha256, str) and len(mission_sha256) == 64
                and all(c in '0123456789abcdef' for c in mission_sha256),
                'Invalid PX4 mission hash')
        self._event('px4_mission_prepared', {'mission_sha256': mission_sha256,
                                             'sent_to_vehicle': False})

    def monitor(self, *, now_ms, target_present=True, map_revision=None,
                envelope_revision=None, return_reserve_ok=True, operator_action=None):
        self._at('EXECUTING')
        require(type(now_ms) is int and now_ms >= 0, 'Invalid monitor time')
        require(operator_action in (None, 'abort', 'return', 'manual'),
                'Unknown operator action')
        reason = None
        if operator_action in ('abort', 'return', 'manual'):
            reason = 'operator_' + operator_action
        elif now_ms >= self.context['expires_at_ms']:
            reason = 'context_expired'
        elif target_present is not True:
            reason = 'target_lost'
        elif map_revision is not None and map_revision != self.context['map_revision']:
            reason = 'map_changed'
        elif envelope_revision is not None and envelope_revision != self.context['envelope_revision']:
            reason = 'envelope_changed'
        elif return_reserve_ok is not True:
            reason = 'return_reserve_reached'
        if reason:
            self.state = 'ABORTED'
            self._event('execution_aborted', {'reason': reason})
            return False
        self._event('execution_monitor', {'now_ms': now_ms, 'passed': True})
        return True

    def capture_result(self, result):
        self._at('EXECUTING')
        self.result = result
        self.state = 'RESULT_CAPTURED'
        self._event('result_captured', {'result_sha256': canonical_sha256(result)})

    def detector_handoff(self, *, detection_evidence_sha256, tracking_controller_revision):
        self._at('EXECUTING')
        require(self.context['mission_profile'] == 'float_recovery_v1',
                'Detector handoff is only valid for drifting-instrument recovery')
        require(isinstance(detection_evidence_sha256, str)
                and len(detection_evidence_sha256) == 64
                and all(c in '0123456789abcdef' for c in detection_evidence_sha256),
                'Invalid detector evidence hash')
        require(isinstance(tracking_controller_revision, str)
                and tracking_controller_revision, 'Missing tracking controller revision')
        self.state = 'DETECTOR_HANDOFF'
        self._event('detector_handoff', {
            'detection_evidence_sha256': detection_evidence_sha256,
            'tracking_controller_revision': tracking_controller_revision,
            'generator_called_in_tracking_loop': False,
        })

    def verify_result(self, images, *, now_ms):
        self._at('RESULT_CAPTURED')
        verified = verify_observation_result(self.result, self.compiled, self.context, images,
                                             now_ms=now_ms)
        self.result = verified
        self.state = 'RESULT_VERIFIED'
        self._event('result_verified', {'result_revision': verified['result_revision'],
                                        'completed': verified['completed']})
        if verified['completed']:
            self.state = 'COMPLETE'
            self._event('complete', {'decision_source': 'tancho_model'})
        return verified

    def reobserve(self, context, evidence, *, now_ms):
        self._at('RESULT_VERIFIED')
        require(self.result['completed'] is False, 'Completed mission cannot be reobserved')
        require(self.cycles < self.context['policy']['max_reobservation_cycles'],
                'Reobservation cycle limit reached')
        require(context['mission_id'] == self.context['mission_id']
                and context['context_id'] != self.context['context_id'],
                'Reobservation requires a new context for the same mission')
        validate_context_v1(context, now_ms=now_ms, evidence=evidence)
        self.cycles += 1
        self.context = context
        self.intents = self.program = self.compiled = self.result = None
        self.state = 'CONTEXT_VALIDATED'
        self._event('reobservation_context_validated', {'context_id': context['context_id'],
                                                        'cycle': self.cycles,
                                                        'input_sha256': context['input_sha256']})

    def record_fallback(self, fallback_revision, *, reason):
        require(self.state not in TERMINAL, 'Terminal runtime cannot use fallback')
        require(isinstance(fallback_revision, str) and fallback_revision, 'Missing fallback revision')
        self._event('fallback_selected', {'decision_source': 'deterministic_fallback',
                                          'fallback_revision': fallback_revision,
                                          'reason': reason,
                                          'counts_as_model_success': False})

    def abort(self, reason):
        require(self.state not in TERMINAL, 'Runtime already terminal')
        self.state = 'ABORTED'
        self._event('aborted', {'reason': reason})

    def audit_record(self):
        require(bool(self.audit), 'Empty audit')
        return {'schema_version': 'tancho-observation-audit-1.0',
                'mission_id': self.context['mission_id'], 'state': self.state,
                'cycles': self.cycles, 'events': list(self.audit),
                'audit_head': self.audit[-1]['event_sha256']}
