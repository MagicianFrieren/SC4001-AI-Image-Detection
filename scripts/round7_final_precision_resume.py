"""Documented numerical-only execution addendum; original freeze/code stay unchanged."""
import sys
sys.dont_write_bytecode = True
import argparse
import numpy as np
from scipy.special import expit
import round7_final_common as f
from round7_final_common import c

ADDENDUM=f.ROOT/'results/round7/final_execution_addendum.json'


def calibrate_float64(raw,temperature):
    # Match Step C's float(row['logit']) / saved_temperature reporting arithmetic.
    return np.asarray(raw,dtype=np.float64)/float(temperature)


def validate_addendum():
    value=c.read(ADDENDUM)
    c.require(value['original_freeze_sha256']==c.sha(f.FREEZE),'Original freeze changed')
    for name,digest in value['files_sha256'].items():
        c.require(c.sha(f.ROOT/name)==digest,f'Execution addendum file changed: {name}')
    for name,digest in value['notebook_sources_sha256'].items():
        c.require(f.notebook_source_hash(f.ROOT/name)==digest,'Resume notebook source changed')
    for name,digest in value['preserved_completed_outputs_sha256'].items():
        c.require(c.sha(f.ROOT/name)==digest,'Previously completed inference bytes changed')
    return value


def install_numerical_adapter():
    import round7_final_models as model_code
    original=model_code.resnet_logits
    def restored_reporting_precision(task,models,condition,records,guard,freeze):
        outputs=original(task,models,condition,records,guard,freeze)
        raw_model=next(r for r in models if r['calibration']=='raw' and r['temperature']==1.)
        raw=outputs[raw_model['id']][0]
        # GPU forward/raw caches remain FP32. Only saved-temperature division and
        # subsequent sigmoid are promoted to the reporting precision used in Step C.
        return {r['id']:(calibrate_float64(raw,r['temperature']),None) for r in models}
    model_code.resnet_logits=restored_reporting_precision


def self_test():
    raw=np.linspace(-20,20,1001,dtype=np.float32)
    for temperature in (.2,1.,4.3):
        actual=calibrate_float64(raw,temperature)
        expected=np.array([float(z)/temperature for z in raw],dtype=np.float64)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(expit(actual),np.array([float(expit(float(z))) for z in expected]))
    # The raw-model logit values are preserved exactly by promotion.
    np.testing.assert_array_equal(calibrate_float64(raw,1.),raw.astype(np.float64))
    print('Precision adapter tests passed: exact Step C reporting arithmetic; raw logits unchanged.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test(); return
    validate_addendum()
    if args.verify:
        from round7_final_verify import verify
        result=verify()
        result.update(execution_addendum_sha256=c.sha(ADDENDUM),execution_addendum_validation='passed')
        f.publish(f.OUT/'verification_precision.json',f.json_bytes(result),resume=True)
        print(c.json.dumps(result,indent=2))
    else:
        c.require(args.resume,'Only continuation of the existing frozen run is permitted')
        install_numerical_adapter()
        from round7_final_run import execute
        execute(True)


if __name__=='__main__':
    main()
