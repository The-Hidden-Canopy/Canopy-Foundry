import unittest
from scripts.neural_forge_worker import HubClient, bounded_event, public_event

class ProjectionTests(unittest.TestCase):
    def test_private_strings_cannot_escape_through_numeric_fields(self):
        projected = public_event({'type':'step', 'loss':'SYNTHETIC_PRIVATE_PATH',
                                  'step':'SYNTHETIC_SECRET', 'status':'SYNTHETIC_SECRET',
                                  'checkpoint_written':'SYNTHETIC_SECRET',
                                  'message':'SYNTHETIC_PRIVATE_PATH'})
        self.assertEqual(projected, {'type':'step'})

    def test_unknown_labels_and_nonfinite_values_do_not_escape(self):
        projected = public_event({'type':'SYNTHETIC_SECRET', 'status':'SYNTHETIC_SECRET',
                                  'loss':float('nan'), 'learning_rate':float('inf'),
                                  'step':True, 'checkpoint_written':1})
        self.assertEqual(projected, {'type':'native_event'})

    def test_update_projects_before_mocked_transport(self):
        captured = {}
        client = object.__new__(HubClient)
        client._request = lambda method, path, body: captured.update(method=method, path=path, body=body)
        client.update('nf-12345678', status='running', operation_id='nfu-synthetic-test',
                      event={'type':'step', 'step':'SYNTHETIC_SECRET',
                             'loss':'SYNTHETIC_PRIVATE_PATH', 'step_private':'SYNTHETIC_SECRET'})
        self.assertEqual(captured['method'], 'POST')
        self.assertEqual(captured['body']['event'], {'type':'step'})
        self.assertEqual(captured['body']['status'], 'running')
        self.assertEqual(captured['body']['operation_id'], 'nfu-synthetic-test')

    def test_opencl_self_check_preserves_only_public_label_and_status(self):
        self.assertEqual(public_event({'type':'self_check', 'status':'passed',
                                       'device':'SYNTHETIC_PRIVATE_DEVICE'}),
                         {'type':'self_check', 'status':'passed'})

    def test_supported_observed_values_survive(self):
        observed = {'type':'step', 'status':'running', 'step':3, 'loss':0.5,
                    'checkpoint_written':False}
        self.assertEqual(public_event(observed), observed)
        self.assertEqual(bounded_event({'type':'step','message':'x'*400})['message'], 'x'*256)

if __name__ == '__main__': unittest.main()
