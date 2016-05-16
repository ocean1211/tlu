import numpy as np
from sys import argv

class TLU():
    def __init__(self, tlu_type, weight_vector):
        self.weights = weight_vector
        self.tlu_class = tlu_type

    def activation(self, in_vector):
        return sum(self.weights * in_vector)

    def activity(self, in_vector):
        return 1/(1+np.e**(-self.activation(in_vector)))
    
    def update_weights(self, weight_vector):
        self.weights = weight_vector

def split_dataset(dataset):

    classes    = np.array(dataset[:,:1])
    inputs     = np.concatenate((np.ones((len(dataset), 1)), dataset[:,1:]), axis=1) #augment data with column of ones
    dim_inputs = len(dataset[:, 1:]) #dimensionality prior to augmentation
    
    return classes, inputs, dim_inputs

def train(tlu, classes, inputs, alpha):        

    weights         = tlu.weights    
    target_min      = 0.1
    target_max      = 0.9        
    threshold       = 0.5
    training_errors = 0
    
    for i in range(len(inputs)):
        activity = tlu.activity(inputs[i])
        
        if classes[i] != tlu.tlu_class:
            target = target_min
        else:
            target = target_max
            if activity < threshold:
                training_errors = training_errors + 1
                
        for j in range(len(tlu.weights)):
            weights[j] = weights[j] + alpha*(target - activity)*inputs[i][j]

        tlu.update_weights(weights)
        
    return training_errors
    
def test (tlu, classes, inputs):

    test_errors = 0
    threshold   = 0.5

    for i in range(len(inputs)):
        activity = tlu.activity(inputs[i])

        if ((activity > threshold) and (tlu.tlu_class != classes[i])) or ((activity < threshold) and (tlu.tlu_class == classes[i])):
            test_errors = test_errors + 1
    
    return test_errors

if __name__ == "__main__":
    _, trainFile, validationFile, modelFile = argv

    try:
        train_set    = np.loadtxt(trainFile)
        validate_set = np.loadtxt(validationFile)
    except IOError:
        print("The datasets failed to load!")
        exit
        
    train_classes, train_inputs, dim_inputs = split_dataset(train_set)
    validate_classes, validate_inputs, _    = split_dataset(validate_set)

    np.random.seed(13579) #intentionally reproducible!
    init_weights = 2*(1/np.sqrt(1+dim_inputs)) * np.random.random_sample((len(train_inputs[0]),)) - (1/np.sqrt(1+dim_inputs))

    TLU_list = []

    num_TLUs = np.unique(train_classes)
    
    for k in range(len(num_TLUs)):
        TLU_list.append(TLU(num_TLUs[k], np.copy(init_weights)))

    repeated_error   = 0 #counter for times lowest error is repeated
    min_v_error_rate = None
    
    error_array = []

    for epoch in range(100000):
        alpha = 1/(10+epoch**2) #where t is epoch

        #epoch trtlu0 trtlu1 ... vrtlu0 vrtlu1 ...
    
        error_vector = [epoch]
        train_error_vector = []
        validate_error_vector = []
    
        for i in range(len(TLU_list)):
            train_error_vector.extend([train(TLU_list[i], train_classes, train_inputs, alpha)/len(train_inputs)])
            validate_error_vector.extend([test(TLU_list[i], validate_classes, validate_inputs)/len(validate_inputs)])

        # length(TLU list) - 1 to account for the division that took place in the loop above
        train_error_vector.extend([sum(train_error_vector)/(len(TLU_list)-1)]) 
        validate_error_vector.extend([sum(validate_error_vector)/(len(TLU_list)-1)])
    
        error_vector.extend(train_error_vector)
        error_vector.extend(validate_error_vector)
        error_array.append(error_vector)

        #test if the error rate becomes repeated
        #start a counter if it does, reset counter if it does not
        v_error_rate = validate_error_vector[0]
    
        if epoch == 0:
            min_v_error_rate = v_error_rate
            continue
    
        if v_error_rate < min_v_error_rate:
            min_v_error_rate = v_error_rate
            repeated_error = 0
        else:
            repeated_error = repeated_error + 1
    
        if repeated_error >= 5:
            break

    error_array = np.array(error_array)
        
    weights = []

    for i in range(len(TLU_list)):
        weight_row = np.hstack([TLU_list[i].tlu_class, TLU_list[i].weights]).tolist()
        weights.append(weight_row)

    weights = np.array(weights)

    np.savetxt(modelFile, weights)
    np.savetxt("errorFile_" + trainFile, error_array)
