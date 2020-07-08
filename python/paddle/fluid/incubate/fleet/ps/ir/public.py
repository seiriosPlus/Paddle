# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright(c) 2020 PaddlePaddle Authors.All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import collections
import math
import os

import six
from paddle.fluid import core
from paddle.fluid.core import CommContext
from paddle.fluid.incubate.fleet.ps.mode import DistributedMode
from paddle.fluid.incubate.fleet.ps.ir import vars_metatools
from paddle.fluid.incubate.fleet.ps.ir.ps_dispatcher import RoundRobin, PSDispatcher

OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "@CLIP"
STEP_COUNTER = "@PS_STEP_COUNTER@"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize


def _get_lr_ops(program):
    lr_ops = []
    for index, op in enumerate(program.global_block().ops):
        role_id = int(op.attr(RPC_OP_ROLE_ATTR_NAME))
        if role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) or \
                        role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) | \
                        int(OPT_OP_ROLE_ATTR_VALUE):
            lr_ops.append(op)
    return lr_ops


def is_sparse_op(op):
    if op.type == "lookup_table" and op.attr('is_sparse') is True:
        return True

    if op.type == "distributed_lookup_table":
        return True

    return False


def get_sparse_tablename(op):
    if is_sparse_op(op):
        return op.input("W")[0]
    return None


def get_all_get_sparse_tablenames(program):
    tablenames = set()
    for op in program.global_block().ops:
        varname = get_sparse_tablename(op)
        if varname:
            tablenames.add(varname)
    return list(tablenames)


def pretty_print_envs(envs, header=None):
    spacing = 5
    max_k = 45
    max_v = 20

    for k, v in envs.items():
        max_k = max(max_k, len(k))
        max_v = max(max_v, len(str(v)))

    h_format = "{{:^{}s}}{}{{:<{}s}}\n".format(max_k, " " * spacing, max_v)
    l_format = "{{:<{}s}}{{}}{{:<{}s}}\n".format(max_k, max_v)
    length = max_k + max_v + spacing

    border = "".join(["="] * length)
    line = "".join(["-"] * length)

    draws = ""
    draws += border + "\n"

    if header and isinstance(header, tuple):
        draws += h_format.format(header[0], header[1])
    else:
        draws += h_format.format("Global Envs", "Value")

    draws += line + "\n"

    for k, v in envs.items():
        draws += l_format.format(k, " " * spacing, str(v))

    draws += border

    _str = "\n{}\n".format(draws)
    return _str


class ServerRuntimeConfig(object):
    def __init__(self):
        self._rpc_send_thread_num = int(
            os.getenv("FLAGS_rpc_send_thread_num", "12"))
        self._rpc_get_thread_num = int(
            os.getenv("FLAGS_rpc_get_thread_num", "12"))
        self._rpc_prefetch_thread_num = int(
            os.getenv("FLAGS_rpc_prefetch_thread_num", "12"))


class MergedVariable:
    def __init__(self, merged, ordered, offsets):
        self.merged_var = merged
        self.ordered_vars = ordered
        self.offsets = offsets

    def __str__(self):
        ordered_varnames = ",".join([var.name for var in self.ordered_vars])
        return "merged: {}\norderd: {}\n".format(self.merged_var.name,
                                                 ordered_varnames)


class CompileTimeStrategy(object):
    def __init__(self, main_program, startup_program, strategy, role_maker):

        self.min_block_size = 8192

        self.origin_main_program = main_program
        self.origin_startup_program = startup_program

        self.strategy = strategy
        self.role_maker = role_maker

        self.origin_sparse_pairs = []
        self.origin_dense_pairs = []

        self.merged_variables_pairs = []
        self.merged_dense_pairs = []
        self.merged_sparse_pairs = []

        self.merged_variable_map = {}
        self.param_name_to_grad_name = {}
        self.grad_name_to_param_name = {}

        self.param_grad_ep_mapping = collections.OrderedDict()
        self.grad_param_mapping = collections.OrderedDict()

        self._build_var_distributed()

    def get_distributed_mode(self):
        trainer = self.strategy.get_trainer_runtime_config()
        return trainer.mode

    def is_sync_mode(self):
        trainer = self.strategy.get_trainer_runtime_config()
        return trainer.mode == DistributedMode.SYNC

    def is_geo_mode(self):
        trainer = self.strategy.get_trainer_runtime_config()
        return trainer.mode == DistributedMode.GEO

    def is_async_mode(self):
        trainer = self.strategy.get_trainer_runtime_config()
        return trainer.mode == DistributedMode.ASYNC

    def get_role_id(self):
        return self.role_maker.role_id()

    def get_trainers(self):
        return self.role_maker.worker_num()

    def get_ps_endpoint(self):
        return self.role_maker.get_pserver_endpoints()[self.get_role_id()]

    def get_ps_endpoints(self):
        return self.role_maker.get_pserver_endpoints()

    def get_origin_programs(self):
        return self.origin_main_program, self.origin_startup_program

    def get_origin_main_program(self):
        return self.origin_main_program

    def get_origin_startup_program(self):
        return self.origin_startup_program

    def get_sparse_varname_on_ps(self, endpoint=None):
        if not endpoint:
            endpoint = self.get_ps_endpoint()

        varnames = get_all_get_sparse_tablenames(self.get_origin_main_program())

        ps_sparse_varnames = []

        for varname in varnames:
            tables = self.get_var_distributed(varname, True)

            for i in range(len(tables)):
                table, ep, _ = tables[i]
                if ep == endpoint:
                    ps_sparse_varnames.append(table)
        return ps_sparse_varnames

    def build_ctx(self, vars, mapping, is_grad, is_sparse, is_send):
        def get_grad_var_ep(slices):
            names = []
            eps = []
            sections = []

            for slice in slices:
                if self.is_geo_mode():
                    if is_send:
                        names.append("{}.delta".format(slice.name))
                    else:
                        names.append(slice.name)
                elif is_grad and self.is_sync_mode() and self.get_trainers(
                ) > 1:
                    names.append("{}.trainer_{}".format(slice.name,
                                                        self.get_role_id()))
                else:
                    names.append(slice.name)

                sections.append(slice.shape[0])

                for ep, pairs in self.param_grad_ep_mapping.items():
                    params, grads = pairs["params"], pairs["grads"]

                    for var in params + grads:
                        if slice.name == var.name:
                            eps.append(ep)
                            break
            return names, eps, sections

        if isinstance(vars, MergedVariable):
            name = vars.merged_var.name
            slices = mapping[name]
            names, eps, sections = get_grad_var_ep(slices)
            origin_varnames = [var.name for var in vars.ordered_vars]
        else:
            name = vars.name
            slices = mapping[name]
            names, eps, sections = get_grad_var_ep(slices)
            origin_varnames = [vars.name]

        trainer_id = self.get_role_id()
        aggregate = True
        ctx = CommContext(name, names, eps, sections, origin_varnames,
                          trainer_id, aggregate, is_sparse)
        return ctx

    def get_trainer_send_context(self):
        send_ctx = {}
        if not self.is_geo_mode():
            for merged in self.merged_dense_pairs:
                grad = merged[1]
                ctx = self.build_ctx(grad, self.grad_var_mapping, True, False,
                                     True)
                send_ctx[ctx.var_name()] = ctx

            for merged in self.merged_sparse_pairs:
                grad = merged[1]
                ctx = self.build_ctx(grad, self.grad_var_mapping, True, True,
                                     True)
                send_ctx[ctx.var_name()] = ctx

            if self.is_async_mode():
                name, ctx = self._step_ctx()
                send_ctx[name] = ctx
        else:
            for pairs in self.origin_sparse_pairs:
                param, grad = pairs
                param_ctx = self.build_ctx(param, self.param_var_mapping, False,
                                           True, True)
                grad_ctx = self.build_ctx(grad, self.grad_var_mapping, True,
                                          True, True)

                ctx = CommContext(param_ctx.var_name(),
                                  param_ctx.split_varnames(),
                                  param_ctx.split_endpoints(),
                                  param_ctx.sections(),
                                  grad_ctx.origin_varnames(),
                                  param_ctx.trainer_id(),
                                  param_ctx.aggregate(), param_ctx.is_sparse())

                send_ctx[ctx.var_name()] = ctx
            name, ctx = self._step_ctx()
            send_ctx[name] = ctx
        return send_ctx

    def get_communicator_send_context(self):
        send_ctx = {}
        if self.is_geo_mode():
            for pairs in self.merged_dense_pairs:
                param = pairs[0]
                ctx = self.build_ctx(param, self.param_var_mapping, False,
                                     False, True)
                send_ctx[ctx.var_name()] = ctx

            for pairs in self.merged_sparse_pairs:
                param = pairs[0]
                ctx = self.build_ctx(param, self.param_var_mapping, False, True,
                                     True)
                send_ctx[ctx.var_name()] = ctx
            name, ctx = self._step_ctx()
            send_ctx[name] = ctx
        else:
            for merged in self.merged_dense_pairs:
                grad = merged[1]
                ctx = self.build_ctx(grad, self.grad_var_mapping, True, False,
                                     True)
                send_ctx[ctx.var_name()] = ctx

            for merged in self.merged_sparse_pairs:
                grad = merged[1]
                ctx = self.build_ctx(grad, self.grad_var_mapping, True, False,
                                     True)
                send_ctx[ctx.var_name()] = ctx

            name, ctx = self._step_ctx()
            send_ctx[name] = ctx
        return send_ctx

    def get_communicator_recv_context(self, recv_type=1):
        # recv_type
        # 1 : DENSE 2. SPARSE 3. ALL
        sparse_varnames = []

        for pairs in self.origin_sparse_pairs:
            param, grad = pairs
            sparse_varnames.append(param.name)

        dense_recv_ctx = {}
        sparse_recv_ctx = {}

        for merged in self.merged_variables_pairs:
            params = merged[0]
            if params.merged_var.name in sparse_varnames:
                continue

            ctx = self.build_ctx(params, self.param_var_mapping, False, False,
                                 False)
            dense_recv_ctx[ctx.var_name()] = ctx

        for pairs in self.origin_sparse_pairs:
            param, grad = pairs
            ctx = self.build_ctx(param, self.param_var_mapping, False, True,
                                 False)
            sparse_recv_ctx[ctx.var_name()] = ctx

        if recv_type == 1:
            return dense_recv_ctx
        if recv_type == 2:
            return sparse_recv_ctx
        if recv_type == 3:
            dense_recv_ctx.update(sparse_recv_ctx)
            return dense_recv_ctx
        assert ValueError("recv_type can only be 1/2/3")

    def get_server_runtime_config(self):
        return self.strategy.get_server_runtime_config()

    def get_var_distributed(self, varname, is_param):
        var_distributed = []
        offset = 0
        if is_param:
            params = self.param_var_mapping[varname]
            param_varnames = [var.name for var in params]
            for ep, pairs in self.param_grad_ep_mapping.items():
                for p in pairs["params"]:
                    if p.name in param_varnames:
                        offset += p.shape[0]
                        var_distributed.append((p.name, ep, p.shape[0]))
        else:
            grads = self.grad_var_mapping[varname]
            grad_varnames = [var.name for var in grads]
            for ep, pairs in self.param_grad_ep_mapping.items():
                for g in pairs["grads"]:
                    if g.name in grad_varnames:
                        var_distributed.append((g.name, ep, g.shape[0]))
        return var_distributed

    def display(self):
        header = ("Fleet Compiled Config", "Value")
        maps = {}

        for ep, pairs in self.param_grad_ep_mapping.items():

            vs = []

            params, grads = pairs

            vs.append("P: ")
            for p in params:
                vs.append(p.name)

            vs.append(", G: ")
            for g in grads:
                vs.append(g.name)

            maps[ep] = " ".join(vs)

        pretty_print_envs(maps, header)

    def _step_ctx(self):
        name = STEP_COUNTER
        trainer_id = self.get_role_id()
        endpoints = self.get_ps_endpoints()
        sections = [1] * len(endpoints)
        names = [name] * len(endpoints)
        ctx = CommContext(name, names, endpoints, sections, [name], trainer_id,
                          True, False)
        return name, ctx

    def _create_vars_from_blocklist(self, block_list):
        """
        Create vars for each split.
        NOTE: only grads need to be named for different trainers, use
              add_trainer_suffix to rename the grad vars.
        Args:
            block_list (list[(varname, block_id, block_size)]): List of gradient blocks.
            add_trainer_suffix (Bool): Add trainer suffix to new variable's name if set True.
        Returns:
            var_mapping (collections.OrderedDict(varname->[new_varname_variable])):A dict mapping
                from original var name to each var split.
        """

        # varname->[(block_id, current_block_size)]
        block_map = collections.OrderedDict()
        var_mapping = collections.OrderedDict()

        for block_str in block_list:
            varname, offset, size = block_str.split(":")
            if varname not in block_map:
                block_map[varname] = []
            block_map[varname].append((int(offset), int(size)))

        for varname, split in six.iteritems(block_map):
            orig_var = self.merged_variable_map[varname]

            if len(split) == 1:
                var_mapping[varname] = [orig_var]
                self.var_distributed.add_distributed_var(
                    origin_var=orig_var,
                    slice_var=orig_var,
                    block_id=0,
                    offset=0,
                    is_slice=False,
                    vtype="Param")
            else:
                var_mapping[varname] = []
                orig_shape = orig_var.shape
                orig_dim1_flatten = 1

                if len(orig_shape) >= 2:
                    orig_dim1_flatten = reduce(lambda x, y: x * y,
                                               orig_shape[1:])

                for i, block in enumerate(split):
                    size = block[1]
                    rows = size // orig_dim1_flatten
                    splited_shape = [rows]
                    if len(orig_shape) >= 2:
                        splited_shape.extend(orig_shape[1:])

                    new_var_name = "%s.block%d" % (varname, i)
                    slice_var = vars_metatools.VarStruct(
                        name=new_var_name,
                        shape=splited_shape,
                        dtype=orig_var.dtype,
                        type=orig_var.type,
                        lod_level=orig_var.lod_level,
                        persistable=False)
                    var_mapping[varname].append(slice_var)

                    self.var_distributed.add_distributed_var(
                        origin_var=orig_var,
                        slice_var=slice_var,
                        block_id=i,
                        offset=-1,
                        is_slice=False,
                        vtype="Param")

        return var_mapping

    def _dispatcher(self):
        ps_dispatcher = RoundRobin(self.get_ps_endpoints())
        ps_dispatcher.reset()
        grad_var_mapping_items = list(six.iteritems(self.grad_var_mapping))

        sparse_gradnames = [grad.name for _, grad in self.origin_sparse_pairs]

        for grad_varname, splited_vars in grad_var_mapping_items:
            if grad_varname in sparse_gradnames:
                continue

            send_vars = []
            for _, var in enumerate(splited_vars):
                send_vars.append(var)

            recv_vars = []
            for _, var in enumerate(send_vars):
                recv_vars.append(self.grad_param_mapping[var])

            eps = ps_dispatcher.dispatch(recv_vars)

            for i, ep in enumerate(eps):
                self.param_grad_ep_mapping[ep]["params"].append(recv_vars[i])
                self.param_grad_ep_mapping[ep]["grads"].append(send_vars[i])

        for grad_varname, splited_vars in grad_var_mapping_items:
            if grad_varname not in sparse_gradnames:
                continue

            ps_dispatcher.reset()

            send_vars = []
            for _, var in enumerate(splited_vars):
                send_vars.append(var)

            recv_vars = []
            for _, var in enumerate(send_vars):
                recv_vars.append(self.grad_param_mapping[var])

            eps = ps_dispatcher.dispatch(recv_vars)

            for i, ep in enumerate(eps):
                self.param_grad_ep_mapping[ep]["params"].append(recv_vars[i])
                self.param_grad_ep_mapping[ep]["grads"].append(send_vars[i])

    def _slice_variable(self,
                        var_list,
                        slice_count,
                        min_block_size,
                        uniform=False):
        """
        We may need to split dense tensor to one or more blocks and put
        them equally onto parameter server. One block is a sub-tensor
        aligned by dim[0] of the tensor.

        We need to have a minimal block size so that the calculations in
        the parameter server side can gain better performance. By default
        minimum block size 8K elements (maybe 16bit or 32bit or 64bit).

        Args:
            var_list (list): List of variables.
            slice_count (int): Numel of count that variables will be sliced, which
                could be the pserver services' count.
            min_block_size (int): Minimum split block size.
        Returns:
            blocks (list[(varname, block_id, current_block_size)]): A list
                of VarBlocks. Each VarBlock specifies a shard of the var.
        """
        blocks = []
        for var in var_list:
            if not uniform:
                var_numel = reduce(lambda x, y: x * y, var.shape)

                if min_block_size == -1:
                    split_count = 1
                else:
                    split_count = slice_count
                    max_pserver_count = int(
                        math.floor(var_numel / float(min_block_size)))
                    if max_pserver_count == 0:
                        max_pserver_count = 1
                    if max_pserver_count < slice_count:
                        split_count = max_pserver_count
                block_size = int(math.ceil(var_numel / float(split_count)))

                if len(var.shape) >= 2:
                    # align by dim1(width)
                    dim1 = reduce(lambda x, y: x * y, var.shape[1:])
                    remains = block_size % dim1
                    if remains != 0:
                        block_size += dim1 - remains
                        # update split_count after aligning
                split_count = int(math.ceil(var_numel / float(block_size)))
                for block_id in range(split_count):
                    curr_block_size = min(block_size, var_numel - (
                        (block_id) * block_size))
                    block = vars_metatools.VarBlock(var.name, block_id,
                                                    curr_block_size)
                    blocks.append(str(block))

            else:
                split_count = slice_count
                numel = reduce(lambda x, y: x * y, var.shape)

                for block_id in range(split_count):
                    block = vars_metatools.VarBlock(var.name, block_id, numel)
                    blocks.append(str(block))
        return blocks

    def _get_param_grad_blocks(self, pairs, min_block_size, uniform=False):
        param_list = []
        grad_list = []
        param_grad_set = set()
        for p, g in pairs:
            # todo(tangwei12) skip parameter marked not trainable
            # if type(p) == Parameter and p.trainable == False:
            # continue
            p = p.merged_var
            g = g.merged_var

            if p.name not in param_grad_set:
                param_list.append(p)
                param_grad_set.add(p.name)
            if g.name not in param_grad_set:
                grad_list.append(g)
                param_grad_set.add(g.name)

                # when we slice var up into blocks, we will slice the var according to
                # pserver services' count. A pserver may have two or more listening ports.
        grad_blocks = self._slice_variable(grad_list,
                                           len(self.get_ps_endpoints()),
                                           min_block_size, uniform)

        param_blocks = self._slice_variable(param_list,
                                            len(self.get_ps_endpoints()),
                                            min_block_size, uniform)
        return param_blocks, grad_blocks

    def _var_slice_and_distribute(self):
        # update these mappings for further transpile:
        # 1. param_var_mapping : param var name->[split params vars]
        # 2. grad_var_mapping : grad var name->[split grads vars]
        # 3. grad_param_mapping : grad.blockx->param.blockx
        # 4. param_grad_ep_mapping : ep->{"params" : [], "grads" : [] }

        dps, dgs = self._get_param_grad_blocks(self.merged_dense_pairs, -1,
                                               False)
        sps, sgs = self._get_param_grad_blocks(self.merged_sparse_pairs,
                                               self.min_block_size, True)

        param_blocks = dps + sps
        grad_blocks = dgs + sgs

        assert (len(grad_blocks) == len(param_blocks))

        # origin_param_name->[splited_param_vars]
        self.param_var_mapping = self._create_vars_from_blocklist(param_blocks)
        self.grad_var_mapping = self._create_vars_from_blocklist(grad_blocks)

        # dict(grad_splited_var->param_splited_var)
        self.grad_param_mapping = collections.OrderedDict()
        for g, p in zip(grad_blocks, param_blocks):
            g_name, g_bid, _ = g.split(":")
            p_name, p_bid, _ = p.split(":")
            self.grad_param_mapping[self.grad_var_mapping[g_name][int(g_bid)]] = \
                self.param_var_mapping[p_name][int(p_bid)]

        print_maps = {}
        for k, v in self.grad_param_mapping.items():
            print_maps[str(k)] = str(v)

        # create mapping of endpoint->split var to create pserver side program
        self.param_grad_ep_mapping = collections.OrderedDict()
        [
            self.param_grad_ep_mapping.update({
                ep: {
                    "params": [],
                    "grads": []
                }
            }) for ep in self.get_ps_endpoints()
        ]

    def _build_var_distributed(self):
        self.var_distributed = vars_metatools.VarsDistributed()

        sparse_pairs, dense_pairs = self.get_param_grads()
        origin_for_sparse = []
        origin_for_dense = []
        param_name_grad_name = dict()
        grad_name_to_param_name = dict()

        for param, grad in sparse_pairs:
            param = vars_metatools.create_var_struct(param)
            grad = vars_metatools.create_var_struct(grad)
            origin_for_sparse.append((param, grad))

        for param, grad in dense_pairs:
            param = vars_metatools.create_var_struct(param)
            grad = vars_metatools.create_var_struct(grad)
            origin_for_dense.append((param, grad))

        for dense_pair in origin_for_dense:
            param, grad = dense_pair

            m_param = MergedVariable(param, [param], [0])
            m_grad = MergedVariable(grad, [grad], [0])
            self.merged_variables_pairs.append((m_param, m_grad))
            self.merged_dense_pairs.append((m_param, m_grad))

        for sparse_pair in origin_for_sparse:
            param, grad = sparse_pair

            m_param = MergedVariable(param, [param], [0])
            m_grad = MergedVariable(grad, [grad], [0])
            self.merged_variables_pairs.append((m_param, m_grad))
            self.merged_sparse_pairs.append((m_param, m_grad))

        for merged in self.merged_variables_pairs:
            m_param, m_grad = merged
            self.merged_variable_map[
                m_param.merged_var.name] = m_param.merged_var
            self.merged_variable_map[m_grad.merged_var.name] = m_grad.merged_var

        param_merges = []
        param_merges.extend(origin_for_sparse)
        param_merges.extend(origin_for_dense)

        for param, grad in param_merges:
            param_name_grad_name[param.name] = grad.name
            grad_name_to_param_name[grad.name] = param.name

        self.origin_sparse_pairs = origin_for_sparse
        self.origin_dense_pairs = origin_for_dense
        self.param_name_to_grad_name = param_name_grad_name
        self.grad_name_to_param_name = grad_name_to_param_name

        sparse_pair_map = collections.OrderedDict()

        for pair in self.origin_sparse_pairs + self.origin_dense_pairs:
            param, grad = pair
            sparse_pair_map[param.name] = str(param)
            sparse_pair_map[grad.name] = str(grad)

        self._var_slice_and_distribute()
        self._dispatcher()

    def dense_var_merge(self, denses):
        if not denses:
            return [], [], None

        ordered_dense = []
        ordered_dense_offsets = []
        flatten_dims = 0

        for dense in denses:
            param, grad = dense

            if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
                raise TypeError(
                    "{} may not Dense Param, need check `build_var_distributed`".
                    format(param.name))

            ordered_dense.append(dense)
            ordered_dense_offsets.append(flatten_dims)
            flatten_dims += reduce(lambda x, y: x * y, param.shape)

        param, grad = denses[0]

        merged_param = vars_metatools.VarStruct(
            "merged.dense_0", (flatten_dims, ), param.dtype, param.type,
            param.lod_level, param.persistable)
        merged_grad = vars_metatools.VarStruct(
            "merged.dense_0@GRAD", (flatten_dims, ), grad.dtype, grad.type,
            grad.lod_level, grad.persistable)

        return ordered_dense, ordered_dense_offsets, merged_param, merged_grad

    def get_param_grads(self):
        origin_program = self.origin_main_program

        def _get_params_grads(sparse_varnames):
            block = origin_program.global_block()

            dense_param_grads = []
            sparse_param_grads = []

            optimize_params = set()
            origin_var_dict = origin_program.global_block().vars
            role_id = int(core.op_proto_and_checker_maker.OpRole.Backward)
            for op in block.ops:
                if _is_opt_role_op(op):
                    # delete clip op from opt_ops when run in Parameter Server mode
                    if OP_NAME_SCOPE in op.all_attrs() \
                            and CLIP_OP_NAME_SCOPE in op.attr(OP_NAME_SCOPE):
                        op._set_attr("op_role", role_id)
                        continue
                    if op.attr(OP_ROLE_VAR_ATTR_NAME):
                        param_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[0]
                        grad_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
                        if param_name not in optimize_params:
                            optimize_params.add(param_name)
                            param_grad = (origin_var_dict[param_name],
                                          origin_var_dict[grad_name])

                            if param_name in sparse_varnames:
                                sparse_param_grads.append(param_grad)
                            else:
                                dense_param_grads.append(param_grad)
            return sparse_param_grads, dense_param_grads

        def _get_sparse_varnames():
            varnames = []
            op_types = {"lookup_table": "W"}
            for op in origin_program.global_block().ops:
                if op.type in op_types.keys() \
                        and op.attr('remote_prefetch') is True:
                    param_name = op.input(op_types[op.type])[0]
                    varnames.append(param_name)

            return list(set(varnames))

        sparse_varnames = _get_sparse_varnames()
        sparse_param_grads, dense_param_grads = _get_params_grads(
            sparse_varnames)

        return sparse_param_grads, dense_param_grads


def _is_opt_role_op(op):
    # NOTE : depend on oprole to find out whether this op is for
    # optimize
    op_maker = core.op_proto_and_checker_maker
    optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
    if op_maker.kOpRoleAttrName() in op.attr_names and \
                    int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(optimize_role):
        return True
    return False


def _get_optimize_ops(_program):
    block = _program.global_block()
    opt_ops = []
    for op in block.ops:
        if _is_opt_role_op(op):
            # delete clip op from opt_ops when run in Parameter Server mode
            if OP_NAME_SCOPE in op.all_attrs() \
                    and CLIP_OP_NAME_SCOPE in op.attr(OP_NAME_SCOPE):
                op._set_attr(
                    "op_role",
                    int(core.op_proto_and_checker_maker.OpRole.Backward))
                continue
            opt_ops.append(op)
    return opt_ops


def _get_varname_parts(varname):
    # returns origin, blockid, trainerid
    orig_var_name = ""
    trainer_part = ""
    block_part = ""
    trainer_idx = varname.find(".trainer_")
    if trainer_idx >= 0:
        trainer_part = varname[trainer_idx + 1:]
    else:
        trainer_idx = len(varname)
    block_index = varname.find(".block")
    if block_index >= 0:
        block_part = varname[block_index + 1:trainer_idx]
    else:
        block_index = len(varname)
    orig_var_name = varname[0:min(block_index, trainer_idx)]
    return orig_var_name, block_part, trainer_part


def _orig_varname(varname):
    orig, _, _ = _get_varname_parts(varname)
    return orig