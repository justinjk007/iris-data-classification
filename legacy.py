def spit_out_name_from_output(output):
    # here _ is the index of the max value. For our output the
    # expected output neuron should be activated the most so the
    # maximum digits index numner should be the actual digit
    max_val, _ = torch.max(output, 0)
    index = _.item()
    if (index == 0):  # the index number
        return "setosa"
    elif (index == 1):
        return "versicolor"
    else:
        return "virginica"

def scale_input(array):
    """Scale input using mean and standard deviation, mean and SD are
     available in the database description"""
    # scaled_input = (input-mean)/standard deviation
    scale_sepal_length = (array[0] - 5.84) / 0.83
    scale_sepal_width = (array[1] - 3.05) / 0.43
    scale_petal_length = (array[2] - 3.76) / 1.76
    scale_petal_width = (array[3] - 1.20) / 0.76
    return [
        scale_sepal_length,
        scale_sepal_width,
        scale_petal_length,
        scale_petal_width,
    ]
