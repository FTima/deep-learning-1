import dnn_blocks as model

def test_initialize_parameters():
    parameters = model.initialize_parameters(4, 3, 1)
    assert (parameters['W1'].shape == (3, 4))
    assert (parameters['b1'].shape == (3, 1))
    assert (parameters['W2'].shape == (1, 3))
    assert (parameters['b2'].shape == (1, 1))

