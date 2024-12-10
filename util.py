

def get_weight_status(model):
    weight_staus = []
    for layer in model.layers:
        if (len(layer.trainable_weights)>0) :
            weight_staus.append(True)
        else :
            weight_staus.append(False)
    return weight_staus

def get_embedding_counts(staus):
    count = 0
    for val in staus :
        if val == True :
            count = count + 1
    return count-1

def model_to_chromosome_size(model) :
    status = get_weight_status(model)
    return get_embedding_counts(status)

def checkFeasible(constraints_specs,current_specs):
    ram = current_specs["ram"] <= constraints_specs["ram"]
    flash = current_specs["flash"] <= constraints_specs["flash"] 
    macc = current_specs["macc"] <= constraints_specs["macc"]
    return ram and flash and macc
