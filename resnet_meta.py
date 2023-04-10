import numpy as np
import pytest
import tvm
#import tvm.testing
from tvm.contrib import graph_executor
from tvm.relay import testing
from tvm import meta_schedule as ms
from tvm import relay, auto_scheduler
from tvm.meta_schedule.testing import relay_workload
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
from tvm.tir.tensor_intrin import *

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
dtype = "float16"

mod, params = testing.resnet.get_workload(
    num_layers=50, batch_size=batch_size, image_shape=image_shape, dtype="float16"
)
#print(mod.astext(show_meta_data=False))
#print(mod)
opt_level = 3
work_dir = "./tune_tmp"
tgt = tvm.target.Target("nvidia/geforce-rtx-3090")
dev = tvm.device(str("cuda"), 0)
target = tvm.target.cuda()

'''
database = ms.relay_integration.tune_relay(
    mod=mod,
    target=tgt,
    work_dir=work_dir,
    max_trials_global=64,
    num_trials_per_iter=1,
    params=params,
    space=ms.space_generator.PostOrderApply(
            sch_rules="cuda-tensorcore", postprocs="cuda-tensorcore", mutator_probs="cuda-tensorcore")
)

rt_mod1 = ms.relay_integration.compile_relay(
    database=database,
    mod=mod,
    target="nvidia/geforce-rtx-3090",
    params=params,
)
print(rt_mod1)
'''
tune_tasks = ms.relay_integration.extract_tasks(mod, tgt, params)
tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
    extracted_tasks=tune_tasks,
    work_dir=work_dir,
    space=ms.space_generator.PostOrderApply(
            sch_rules="cuda-tensorcore", postprocs="cuda-tensorcore", mutator_probs="cuda-tensorcore")
)
database = ms.tune.tune_tasks(
    tasks=tasks,
    task_weights=task_weights,
    work_dir=work_dir,
    max_trials_per_task=4,
    max_trials_global=150,
)
with database, tvm.transform.PassContext(
    opt_level=3,
    config={"relay.backend.use_meta_schedule": True},
):
    lib = relay.build(mod, target=target, params=params)

a_np = np.random.randint(0, 255, size=(1,3,224,224)).astype(dtype)
data = tvm.nd.array(a_np, dev)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("data", data)
module.run()
timer = module.module.time_evaluator("run", dev, number=10, repeat=3)
print(timer())
