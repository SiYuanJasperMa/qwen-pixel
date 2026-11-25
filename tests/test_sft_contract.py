import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel.sft_contract import training_contract, validate_runtime_contract,checkpoint_interface
from pixel.agent_protocol import RGB_SYSTEM,RGB_TOOLS,CHECKED_RGB_SYSTEM,CHECKED_RGB_TOOLS
from pixel.clean_edit_protocol import SYSTEM as CLEAN_SYSTEM,TOOLS as CLEAN_TOOLS
from pixel.sft_contract import infer_runtime_features,signature
import copy


class SFTContractTests(unittest.TestCase):
    def test_clean_runtime_records_numeric_validation_without_rejects_word(self):
        features=infer_runtime_features(CLEAN_SYSTEM,CLEAN_TOOLS)
        self.assertTrue(features['rgb']);self.assertTrue(features['roi']);self.assertTrue(features['checked_rgb'])
        row=self.row(system=CLEAN_SYSTEM);row['tools']=copy.deepcopy(CLEAN_TOOLS)
        with tempfile.TemporaryDirectory()as folder:
            contract=training_contract([row],262144)
            (Path(folder)/'agent_contract.json').write_text(json.dumps(contract))
            wrong={**features,'checked_rgb':False}
            with self.assertRaisesRegex(ValueError,'execution behavior'):
                validate_runtime_contract(folder,CLEAN_SYSTEM,CLEAN_TOOLS,'tool',262144,row['messages'],wrong)

    def test_runtime_file_state_cannot_change_behind_matching_text(self):
        row=self.row(system=RGB_SYSTEM);row['tools']=copy.deepcopy(RGB_TOOLS)
        features=infer_runtime_features(RGB_SYSTEM,RGB_TOOLS)
        other=copy.deepcopy(row);other['runtime_features']={**features,'files_persist':False}
        with self.assertRaisesRegex(ValueError,'Mixed agent'):
            training_contract([row,other],262144)

    def test_legacy_contract_still_uses_original_signature(self):
        row=self.row(system=RGB_SYSTEM);row['tools']=copy.deepcopy(RGB_TOOLS)
        contract=training_contract([row],262144)
        contract.pop('runtime_features')
        contract['signature']=signature(RGB_SYSTEM,RGB_TOOLS,'tool',contract['user_prompt_template'])
        with tempfile.TemporaryDirectory()as folder:
            (Path(folder)/'agent_contract.json').write_text(json.dumps(contract))
            validate_runtime_contract(folder,RGB_SYSTEM,RGB_TOOLS,'tool',262144,row['messages'])
    def test_checkpoint_restore_cannot_remove_numeric_error_feedback(self):
        row=self.row(system=CHECKED_RGB_SYSTEM);row['tools']=copy.deepcopy(CHECKED_RGB_TOOLS)
        with tempfile.TemporaryDirectory()as folder:
            (Path(folder)/'agent_contract.json').write_text(json.dumps(training_contract([row],262144)))
            messages=copy.deepcopy(row['messages']);messages[0]['content']=RGB_SYSTEM
            with self.assertRaisesRegex(ValueError,'numeric validation'):
                checkpoint_interface(folder,messages,RGB_TOOLS)

    def test_sft_target_cannot_teach_multiple_calls_to_single_action_runtime(self):
        row=self.row();row['id']='bad_multi_call'
        row['target_message']={'role':'assistant','tool_calls':[{},{}]}
        with self.assertRaisesRegex(ValueError,'one complete tool action'):
            training_contract([row],262144)
        row['target_message']={'role':'tool','tool_calls':[{}]}
        with self.assertRaisesRegex(ValueError,'assistant action'):
            training_contract([row],262144)

    def test_checkpoint_text_restore_keeps_schema_and_user_checks(self):
        row=self.row(system=RGB_SYSTEM);row['tools']=copy.deepcopy(RGB_TOOLS)
        contract=training_contract([row],262144)
        with tempfile.TemporaryDirectory()as folder:
            (Path(folder)/'agent_contract.json').write_text(json.dumps(contract))
            messages=copy.deepcopy(row['messages']);messages[0]['content']+='\nNew helper descriptions.'
            restored,tools=checkpoint_interface(folder,messages,RGB_TOOLS)
            validate_runtime_contract(folder,restored[0]['content'],tools,'tool',262144,restored)
            restored[1]['content'][1]['text']+='\nNew user guidance.'
            with self.assertRaisesRegex(ValueError,'system/tools'):
                validate_runtime_contract(folder,restored[0]['content'],tools,'tool',262144,restored)
            incompatible=copy.deepcopy(RGB_TOOLS)
            incompatible[0]['function']['parameters']['required'].append('unexpected')
            with self.assertRaisesRegex(ValueError,'different runtime profile'):
                checkpoint_interface(folder,messages,incompatible)
    def row(self, system='Edit RGB.', mode='tool'):
        return {'messages':[{'role':'system','content':system},{'role':'user','edit_instruction':'Change color.',
                 'content':[{'type':'image','image':'source.png'},
                            {'type':'text','text':'Source dimensions: width=16, height=12.\nEDIT: Change color.'}]}],
                'tools':[{'name':'python'}], 'tool_image_mode':mode}

    def test_mixed_training_interfaces_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Mixed agent'):
            training_contract([self.row(),self.row(mode='user')],262144)

    def test_inference_cannot_silently_change_trained_interface_or_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            contract=training_contract([self.row()],262144)
            (path/'agent_contract.json').write_text(json.dumps(contract))
            messages=self.row()['messages']
            validate_runtime_contract(path,'Edit RGB.',[{'name':'python'}],'tool',262144,messages)
            with self.assertRaisesRegex(ValueError,'system/tools'):
                validate_runtime_contract(path,'Different prompt.',[{'name':'python'}],'tool',262144,messages)
            with self.assertRaisesRegex(ValueError,'token budget'):
                validate_runtime_contract(path,'Edit RGB.',[{'name':'python'}],'tool',65536,messages)

    def test_user_guidance_cannot_change_even_with_matching_system_and_tools(self):
        with tempfile.TemporaryDirectory()as folder:
            contract=training_contract([self.row()],262144)
            (Path(folder)/'agent_contract.json').write_text(json.dumps(contract))
            messages=self.row()['messages'];messages[1]['content'][1]['text']+='\nDifferent workflow.'
            with self.assertRaisesRegex(ValueError,'system/tools'):
                validate_runtime_contract(folder,'Edit RGB.',[{'name':'python'}],'tool',262144,messages)


if __name__=='__main__':
    unittest.main()
