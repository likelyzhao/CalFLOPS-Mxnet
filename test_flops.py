import calflops 
from mxnet_paser import mxnet_paser

def main():
    #print(calflops.calConvFlops([3,224,224],[64,112,112],[7,7]))
    #print(calflops.calActivationFlops([64,112,112],[64,112,112]))
    paserer = mxnet_paser()
    #ops = paserer.paser('model/personattr-deploy',80)
    #ops = paserer.paser('model/imagenet1k-resnet-18',00)
    #ops = paserer.paser('model/Inception-BN',126,(1,3,224,224))
    ops = paserer.paser('model/resnet-50',0,(1,3,224,224))
    total_ops = {
        'conv_multipflops':0,
        'conv_addflops':0,
        'conv_compareflops':0,
        'conv_expflops':0,
        'Fc_multipflops' :0,
        'Fc_addflops':0,
        'Fc_compareflops':0,
        'Fc_expflops':0,
        'Pool_multipflops' :0,
        'Pool_addflops':0,
        'Pool_compareflops':0,
        'Pool_expflops':0,
        'Activation_multipflops' :0,
        'Activation_addflops':0,
        'Activation_compareflops':0,
        'Activation_expflops':0,
    }

    for op in ops:
        if op['type'] == 'Convolution':
            print('kernel_size',op['attr']['kernel_size'])
            print('in_shape',op['in_shape'])
            print('out_shape',op['out_shape'])
            print('has_bias',op['attr']["no_bias"])
            print('num_group',op['attr']["num_group"])
            multipflops,addflops,compareflops,expflops =calflops.calConvFlops(op['in_shape'],
                                                                op['out_shape'],
                                                                op['attr']['kernel_size'],
                                                                not op['attr']["no_bias"],
                                                                op['attr']["num_group"])
            print('conv',multipflops)
            total_ops['conv_multipflops']+=multipflops
            total_ops['conv_addflops']+=addflops
            total_ops['conv_compareflops']+=compareflops
            total_ops['conv_expflops']+=expflops
        if op['type'] == 'FullyConnected':
            print('in_shape',op['in_shape'])
            print('out_shape',op['out_shape'])
            multipflops,addflops,compareflops,expflops =calflops.calFcFlops(op['in_shape'],
                                                                op['out_shape'],
                                                                not op['attr']["no_bias"])
            total_ops['Fc_multipflops']+=multipflops
            total_ops['Fc_addflops']+=addflops
            total_ops['Fc_compareflops']+=compareflops
            total_ops['Fc_expflops']+=expflops
        
        if op['type'] == 'Pooling':
            print('kernel_size',op['attr']['kernel_size'])
            print('in_shape',op['in_shape'])
            print('out_shape',op['out_shape'])
            print('pool_type',op['attr']["pool_type"])
            multipflops,addflops,compareflops,expflops =calflops.calPoolingFlops(op['in_shape'],
                                                                op['out_shape'],
                                                                op['attr']['kernel_size'],
                                                                op['attr']["pool_type"])

            total_ops['Pool_multipflops']+=multipflops
            total_ops['Pool_addflops']+=addflops
            total_ops['Pool_compareflops']+=compareflops
            total_ops['Pool_expflops']+=expflops

        if op['type'] == 'Activation':
            print('in_shape',op['in_shape'])
            print('out_shape',op['out_shape'])
            print('act_type',op['attr']["act_type"])
            multipflops,addflops,compareflops,expflops =calflops.calActivationFlops(op['in_shape'],
                                                                op['out_shape'],
                                                                op['attr']["act_type"])

            total_ops['Activation_multipflops']+=multipflops
            total_ops['Activation_addflops']+=addflops
            total_ops['Activation_compareflops']+=compareflops
            total_ops['Activation_expflops']+=expflops
    
    
    print('----total flops---')
    print(total_ops)
    for key in total_ops:
        print('{}: {:.5f} GFlops'.format(key,total_ops[key]/1000000000.0))
    #print('addflops:', str(total_ops[1]/1000000000.0),'Gflops')
    #print('compareflops:',str(total_ops[2]/1000000000.0),'Gflops')
    #print('expflops:',str(total_ops[3]/1000000000.0),'Gflops')


if __name__ == "__main__":
    main()