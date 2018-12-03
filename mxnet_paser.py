import mxnet as mx

def get_children(symbol):
    symbollist = symbol.get_children()
    if len(symbollist) >1:
        for sym in symbol:
            return get_children(sym)
    print(symbollist)
    return symbollist

def get_outputs_shape():
    pass

def False_True_2_bool(str):
    if "False" == str:
        return False
    if "True" == str:
        return True

class mxnet_paser():
    layers = {}
    def paser(self,modelpath,iters,datasize=(1,3,224,112)):
        net, arg_params,aux_params = mx.model.load_checkpoint(modelpath,iters)
        data = mx.sym.Variable('data')

        args = net.list_arguments()
        outs = net.list_outputs()
        auxs = net.list_auxiliary_states()

        arg_shape, out_shape,aux_shape  = net.infer_shape(data=datasize)

        print('---'*10 + 'args' + '---'*10)
        for idx in range(len(args)):
            print(args[idx],arg_shape[idx])
        print('---'*10 + 'outputs' + '---'*10)
        for idx in range(len(outs)):
            print(outs[idx],out_shape[idx])
        print('---'*10 + 'auxs' + '---'*10)
        for idx in range(len(auxs)):
            print(auxs[idx],aux_shape[idx])
        
        #print(get_children(net))
        all_layers = net.get_internals()
        idx =0
        all_ops =[]
        str = net.debug_str()

        for split in str.split('--------------------'):
            if 'Op:' in split:
                op ={}
                opstart = split.find('Op:')+3
                opend = split.find(',',opstart)
                print('optype',split[opstart:opend])
                op['type'] = split[opstart:opend]
                namestart = split.find('Name=')+5
                nameend = split.find('\n', namestart)
                print('opname',split[namestart:nameend])
                op['name'] = split[namestart:nameend]

                all_ops.append(op)
            
            #print(split)

        for op in all_ops:
            print(op['name'])
            print(op['type'])
            op['attr'] ={}
            
            opnow = all_layers[op['name']+'_output']
            if op['type'] == 'Convolution':
                #print(opnow.list_attr()['kernel'])
                op['attr']['kernel_size'] =  [int(p) for p in opnow.list_attr()['kernel'].strip()[1:-1].split(',')]
                if "no_bias" in opnow.list_attr():
                    op['attr']["no_bias"] = False_True_2_bool(opnow.list_attr()["no_bias"])
                else:
                    op['attr']["no_bias"] = False
                if "num_group" in opnow.list_attr():
                    op['attr']["num_group"] = int(opnow.list_attr()["num_group"])
                else:
                    op['attr']["num_group"] = 1

            if op['type'] == 'FullyConnected':
                if "no_bias" in opnow.list_attr():
                    op['attr']["no_bias"] = False_True_2_bool(opnow.list_attr()["no_bias"])
                else:
                    op['attr']["no_bias"] = False

            if op['type'] == 'Pooling':
                if 'kernel' in opnow.list_attr():
                    op['attr']['kernel_size'] =  [int(p) for p in opnow.list_attr()['kernel'].strip()[1:-1].split(',')]
                if "global_pool" in opnow.list_attr():
                    op['attr']["global_pool"] = False_True_2_bool(opnow.list_attr()["global_pool"])
                else:
                    op['attr']["global_pool"] = False
                if "pool_type" in opnow.list_attr():
                    op['attr']["pool_type"] = opnow.list_attr()["pool_type"]
                else:
                    op['attr']["pool_type"] = 'max'
                
                if op['attr']["global_pool"] is True:
                    op['attr']["pool_type"] = 'gop'


            if op['type'] == 'Activation':
                if "act_type" in opnow.list_attr():
                    op['attr']["act_type"] = opnow.list_attr()["act_type"]
                else:
                    op['attr']["act_type"] = 'relu'
                
            

            op_up = opnow.get_children()
            #print(op_up)
            _,out_shape,_ = opnow.infer_shape(data=datasize)
            print('out_shape',out_shape)
            op['out_shape'] = out_shape[0][1:]
            _,out_shape,_ = op_up[0].infer_shape(data=datasize)
            print('in_shape',out_shape)
            op['in_shape'] = out_shape[0][1:]

        return all_ops
