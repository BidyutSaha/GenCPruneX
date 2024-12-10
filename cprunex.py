import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Flatten, Input, Conv2D, BatchNormalization, DepthwiseConv2D,
    GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D,
    Conv1D, DepthwiseConv1D, ReLU, MaxPooling1D, GlobalAveragePooling1D,
    SeparableConv1D, Dropout
)
from tensorflow.keras.models import Model



def prune_dense_layer(layer, x, last_mask, pruning_ratio ,flatten_output_size):
    weights = layer.get_weights()[0]
    bias = layer.get_weights()[1] if layer.use_bias else None
    
    if last_mask is not None :
        if (len(last_mask) == weights.shape[0]) :
            weights = weights[last_mask, :]
        else :
            new_mask = []

            for i in range(flatten_output_size) :
                new_mask.append(True)
            for i in range( weights.shape[0] -flatten_output_size ) :
                new_mask.append(False)
            weights = weights[new_mask, :]

    if pruning_ratio == 0:
        pruned_weights = weights
        pruned_bias = bias if layer.use_bias and bias is not None else []
        last_mask = None
    else:
        weight_l2_norms = np.linalg.norm(weights, axis=0)
        threshold = np.percentile(weight_l2_norms, int(pruning_ratio * 100))
        weight_mask = weight_l2_norms > threshold
        last_mask = weight_mask
        pruned_weights = weights[:, weight_mask]
        pruned_bias = bias[weight_mask] if layer.use_bias and bias is not None else []

    input_shape = (pruned_weights.shape[0],)

    new_layer = Dense(units=pruned_weights.shape[1], activation=layer.activation, use_bias=layer.use_bias, input_shape=input_shape)
    new_layer.build((None, pruned_weights.shape[0]))
    
    if layer.use_bias:
        new_layer.set_weights([pruned_weights, pruned_bias])
    else:
        new_layer.set_weights([pruned_weights])
        
    x = new_layer(x)
    return x, last_mask 

def prune_conv2d_layer(layer, x, last_mask, pruning_ratio):
    weights = layer.get_weights()[0]
    bias = layer.get_weights()[1] if layer.use_bias else None  # Step 1: Check if bias is used

    if last_mask is not None:
        weights = weights[..., last_mask, :]

    if pruning_ratio == 0:
        pruned_weights = weights
        pruned_bias = bias if layer.use_bias and bias is not None else []
        last_mask = None
    else:
        weight_l2_norms = np.sqrt(np.sum(np.square(weights), axis=(0, 1, 2)))
        threshold = np.percentile(weight_l2_norms, int(pruning_ratio * 100))
        weight_mask = weight_l2_norms > threshold
        last_mask = weight_mask
        pruned_weights = weights[..., weight_mask]
        pruned_bias = bias[weight_mask] if layer.use_bias and bias is not None else []

    # Create a new Conv2D layer with the pruned weights
    new_layer = Conv2D(filters=pruned_weights.shape[-1], kernel_size=layer.kernel_size, strides=layer.strides,
                       padding=layer.padding, activation=layer.activation, use_bias=layer.use_bias)
    new_layer.build((None, pruned_weights.shape[0], pruned_weights.shape[1], pruned_weights.shape[2]))

    # Step 2: Set weights conditionally based on bias usage
    if layer.use_bias:
        new_layer.set_weights([pruned_weights, pruned_bias])
    else:
        new_layer.set_weights([pruned_weights])

    x = new_layer(x)
    return x, last_mask

def prune_conv1d_layer(layer, x, last_mask, pruning_ratio):
    weights = layer.get_weights()[0]
    bias = layer.get_weights()[1] if layer.use_bias else None  # Step 1: Check if bias is used

    if last_mask is not None:
        weights = weights[:, last_mask, :]

    if pruning_ratio == 0:
        pruned_weights = weights
        pruned_bias = bias if layer.use_bias and bias is not None else []
        last_mask = None
    else:
        weight_l2_norms = np.sqrt(np.sum(np.square(weights), axis=(0, 1)))
        threshold = np.percentile(weight_l2_norms, int(pruning_ratio * 100))
        weight_mask = weight_l2_norms > threshold
        last_mask = weight_mask
        pruned_weights = weights[..., weight_mask]
        pruned_bias = bias[weight_mask] if layer.use_bias and bias is not None else []

    # Create a new Conv1D layer with the pruned weights
    new_layer = Conv1D(filters=pruned_weights.shape[-1], kernel_size=layer.kernel_size, strides=layer.strides,
                       padding=layer.padding, activation=layer.activation, use_bias=layer.use_bias)
    new_layer.build((None, pruned_weights.shape[0], pruned_weights.shape[1]))

    # Step 2: Set weights conditionally based on bias usage
    if layer.use_bias:
        new_layer.set_weights([pruned_weights, pruned_bias])
    else:
        new_layer.set_weights([pruned_weights])

    x = new_layer(x)
    return x, last_mask

def prune_depthwise_conv2d_layer(layer, x, last_mask):
    # Generally no pruning will done in this layer
    
    weights = layer.get_weights()[0]
    bias = layer.get_weights()[1] if len(layer.get_weights()) > 1 else None
    if last_mask is not None:
        weights = weights[..., last_mask, :]
        if bias is not None:
            bias = bias[last_mask]
            
    new_layer = DepthwiseConv2D(kernel_size=layer.kernel_size, strides=layer.strides, padding=layer.padding, activation=layer.activation, 
                                use_bias=layer.use_bias)
    new_layer.build((None, weights.shape[0], weights.shape[1], weights.shape[2]))
    new_layer.set_weights([weights, bias] if bias is not None else [weights])
    x = new_layer(x)
    return x, last_mask

def prune_depthwise_conv1d_layer(layer, x, last_mask):
    # Generally, no pruning will be done in this layer
    
    weights = layer.get_weights()[0]
    bias = layer.get_weights()[1] if len(layer.get_weights()) > 1 else None
    
    # Adjust weights and bias if last_mask is provided
    if last_mask is not None:
        weights = weights[:, last_mask, :]
        if bias is not None:
            bias = bias[last_mask]
            
    # Create a new DepthwiseConv1D layer with the adjusted weights
    new_layer = DepthwiseConv1D(kernel_size=layer.kernel_size, strides=layer.strides, padding=layer.padding, 
                                activation=layer.activation, use_bias=layer.use_bias)
    new_layer.build((None, weights.shape[0], weights.shape[1]))
    
    # Set weights, conditionally adding bias if it exists
    new_layer.set_weights([weights, bias] if bias is not None else [weights])
    x = new_layer(x)
    
    return x, last_mask

def prune_batch_norm_layer(layer, x, last_mask):
    new_layer = BatchNormalization()
    gamma, beta, moving_mean, moving_variance = layer.get_weights()
    if last_mask is not None:
        gamma = gamma[last_mask]
        beta = beta[last_mask]
        moving_mean = moving_mean[last_mask]
        moving_variance = moving_variance[last_mask]
    new_layer.build(x.shape)
    new_layer.set_weights([gamma, beta, moving_mean, moving_variance])
    x = new_layer(x)
    return x, last_mask

def prune_separable_conv2d_layer(layer, x, last_mask, pruning_ratio):
    depthwise_weights = layer.get_weights()[0]
    pointwise_weights = layer.get_weights()[1]
    bias = layer.get_weights()[2] if len(layer.get_weights()) > 2 else None
    
    if last_mask is not None:
        depthwise_weights = depthwise_weights[..., last_mask, :]
        pointwise_weights = pointwise_weights[..., last_mask, :]



    if pruning_ratio == 0:
        depthwise_weights = depthwise_weights
        pruned_pointwise_weights = pointwise_weights  
        pruned_bias = bias
        last_mask = None
        
    else :
        pointwise_l2_norms = np.sqrt(np.sum(np.square(pointwise_weights), axis=(0, 1, 2)))
        threshold = np.percentile(pointwise_l2_norms, int(pruning_ratio * 100))
        pointwise_mask = pointwise_l2_norms > threshold
        last_mask = pointwise_mask
        pruned_pointwise_weights = pointwise_weights[..., pointwise_mask]
        pruned_bias = bias[pointwise_mask] if bias is not None else []
    
        
    new_layer = SeparableConv2D(filters=pruned_pointwise_weights.shape[-1], kernel_size=layer.kernel_size, strides=layer.strides, padding=layer.padding, activation=layer.activation, use_bias=layer.use_bias)
    new_layer.build((None, depthwise_weights.shape[0], depthwise_weights.shape[1], depthwise_weights.shape[2]))
    new_layer.set_weights([depthwise_weights, pruned_pointwise_weights, pruned_bias] if bias is not None else [depthwise_weights, pruned_pointwise_weights])
    x = new_layer(x)
    return x, last_mask

def prune_separable_conv1d_layer(layer, x, last_mask, pruning_ratio):
    depthwise_weights = layer.get_weights()[0]
    pointwise_weights = layer.get_weights()[1]
    bias = layer.get_weights()[2] if len(layer.get_weights()) > 2 else None  # Check for bias

    # Adjust depthwise and pointwise weights if last_mask is provided
    if last_mask is not None:
        depthwise_weights = depthwise_weights[:, last_mask, :]
        pointwise_weights = pointwise_weights[:, last_mask]

    # Prune weights based on the pruning ratio
    if pruning_ratio == 0:
        pruned_pointwise_weights = pointwise_weights
        pruned_bias = bias if bias is not None else []
        last_mask = None
    else:
        # Calculate L2 norms for pruning
        pointwise_l2_norms = np.sqrt(np.sum(np.square(pointwise_weights), axis=(0, 1)))
        threshold = np.percentile(pointwise_l2_norms, int(pruning_ratio * 100))
        pointwise_mask = pointwise_l2_norms > threshold
        last_mask = pointwise_mask
        pruned_pointwise_weights = pointwise_weights[..., pointwise_mask]
        pruned_bias = bias[pointwise_mask] if bias is not None else []

    # Create a new SeparableConv1D layer
    new_layer = SeparableConv1D(filters=pruned_pointwise_weights.shape[-1], kernel_size=layer.kernel_size, strides=layer.strides,
                                padding=layer.padding, activation=layer.activation, use_bias=layer.use_bias)
    new_layer.build((None, depthwise_weights.shape[0], depthwise_weights.shape[1]))

    # Set weights based on whether bias exists
    if bias is not None:
        new_layer.set_weights([depthwise_weights, pruned_pointwise_weights, pruned_bias])
    else:
        new_layer.set_weights([depthwise_weights, pruned_pointwise_weights])

    x = new_layer(x)
    return x, last_mask

def uniform_channel_prune(model, input_shape, pruning_ratio=0.5 , final_decision_layer_idx = -1 , start_index = 1):
    last_mask = None
    new_input_layer = Input(shape=input_shape)
    x = new_input_layer
    flatten_output_size = None
   
    for idx, layer in enumerate(model.layers[start_index:]):
        
        if isinstance(layer, Dense):

            if idx == final_decision_layer_idx -1 :
                pruning_ratio = 0
            x, last_mask = prune_dense_layer(layer, x, last_mask, pruning_ratio , flatten_output_size )
            
            
        elif isinstance(layer, Conv2D):
            x, last_mask = prune_conv2d_layer(layer, x, last_mask, pruning_ratio)
            
        elif isinstance(layer, Conv1D):
            x, last_mask = prune_conv1d_layer(layer, x, last_mask, pruning_ratio)
            
        elif isinstance(layer, DepthwiseConv2D):
            x, last_mask = prune_depthwise_conv2d_layer(layer, x, last_mask)

        elif isinstance(layer, DepthwiseConv1D):
            x, last_mask = prune_depthwise_conv1d_layer(layer, x, last_mask)
            
        elif isinstance(layer, SeparableConv2D):
            x, last_mask = prune_separable_conv2d_layer(layer, x, last_mask, pruning_ratio)

        elif isinstance(layer, SeparableConv1D):
            x, last_mask = prune_separable_conv1d_layer(layer, x, last_mask, pruning_ratio)
            
        elif isinstance(layer, BatchNormalization):
            x, last_mask = prune_batch_norm_layer(layer, x, last_mask)
            
        elif isinstance(layer, GlobalAveragePooling2D):
            x = GlobalAveragePooling2D()(x)

        elif isinstance(layer, GlobalAveragePooling1D):
            x = GlobalAveragePooling1D()(x)
            
        elif isinstance(layer, Flatten):
            x = Flatten()(x)
            flatten_output_size = x.shape[-1]
        
            
        elif isinstance(layer, MaxPooling2D):
            x = MaxPooling2D(pool_size = layer.pool_size, strides = layer.strides )(x)

        elif isinstance(layer, MaxPooling1D):
            x = MaxPooling1D(pool_size = layer.pool_size, strides = layer.strides )(x)

        else:
            x = layer(x)
           
    new_model = Model(inputs=new_input_layer, outputs=x)
    return new_model
 
def custom_channel_prune(model, input_shape, pruning_ratios , start_index = 1):
    last_mask = None
    new_input_layer = Input(shape=input_shape)
    x = new_input_layer
    flatten_output_size = None


    ptr = 0
    for idx, layer in enumerate(model.layers[start_index:]):
        if isinstance(layer, Dense):
            if ptr== len(pruning_ratios) :
                pruning_ratio = 0
            else :
                pruning_ratio = pruning_ratios[ptr]
                ptr = ptr + 1
            
            x, last_mask = prune_dense_layer(layer, x, last_mask, pruning_ratio, flatten_output_size)
            
        elif isinstance(layer, Conv2D):
            if ptr== len(pruning_ratios) :
                pruning_ratio = 0
            else :
                pruning_ratio = pruning_ratios[ptr]
                ptr = ptr + 1
            x, last_mask = prune_conv2d_layer(layer, x, last_mask, pruning_ratio)
            
        elif isinstance(layer, Conv1D):
            if ptr== len(pruning_ratios) :
                pruning_ratio = 0
            else :
                pruning_ratio = pruning_ratios[ptr]
                ptr = ptr + 1
            x, last_mask = prune_conv1d_layer(layer, x, last_mask, pruning_ratio)
            
        elif isinstance(layer, DepthwiseConv2D):
            if ptr== len(pruning_ratios) :
                pruning_ratio = 0
            else :
                pruning_ratio = pruning_ratios[ptr]
                ptr = ptr + 1
            x, last_mask = prune_depthwise_conv2d_layer(layer, x, last_mask)

        elif isinstance(layer, DepthwiseConv1D):
            if ptr== len(pruning_ratios) :
                pruning_ratio = 0
            else :
                pruning_ratio = pruning_ratios[ptr]
                ptr = ptr + 1
            x, last_mask = prune_depthwise_conv1d_layer(layer, x, last_mask)
            
        elif isinstance(layer, SeparableConv2D):
            if ptr== len(pruning_ratios) :
                pruning_ratio = 0
            else :
                pruning_ratio = pruning_ratios[ptr]
                ptr = ptr + 1
            x, last_mask = prune_separable_conv2d_layer(layer, x, last_mask, pruning_ratio)

        elif isinstance(layer, SeparableConv1D):
            if ptr== len(pruning_ratios) :
                pruning_ratio = 0
            else :
                pruning_ratio = pruning_ratios[ptr]
                ptr = ptr + 1
            x, last_mask = prune_separable_conv1d_layer(layer, x, last_mask, pruning_ratio)
            
        elif isinstance(layer, BatchNormalization):
            if ptr== len(pruning_ratios) :
                pruning_ratio = 0
            else :
                pruning_ratio = pruning_ratios[ptr]
                ptr = ptr + 1
            x, last_mask = prune_batch_norm_layer(layer, x, last_mask)
            
        elif isinstance(layer, GlobalAveragePooling2D):
            x = GlobalAveragePooling2D()(x)

        elif isinstance(layer, GlobalAveragePooling1D):
            x = GlobalAveragePooling1D()(x)
            
        elif isinstance(layer, Flatten):
            x = Flatten()(x)
            flatten_output_size = x.shape[-1]
            
        elif isinstance(layer, MaxPooling2D):
            x = MaxPooling2D(pool_size = layer.pool_size, strides = layer.strides )(x)

        elif isinstance(layer, MaxPooling1D):
            x = MaxPooling1D(pool_size = layer.pool_size, strides = layer.strides )(x)

        else:
            x = layer(x)
    new_model = Model(inputs=new_input_layer, outputs=x)
    return new_model