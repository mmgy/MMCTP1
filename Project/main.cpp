#include "nn.h"

int main(int, char**) {

    //The layers and their features must be defined first. Number of inputs, number of outputs and the name of the activation function.
    //Each layer defined that way is a config object, a class I defined in the nn.h header file.
    config layer1{1, 16, "sigmoid"};
    config layer2{16, 64, "relu"};
    config layer3{64, 16, "relu"};
    //config layer4{256, 128, "relu"};
    config layer5{16, 1, "relu"};

    //The architecture of the neural network is built here. Basically it is a vector of config objects.
    std::vector<config> nn_architecture{layer1, layer2, layer3, layer5};

    //Here are the bounds of the MinMaxScaler which scales the data as well as the targets separately.
    double train_min = -500.001;
    double train_max = 500.001;
    double train_resolution = 1;
    double val_min = -1000.001;
    double val_max = 1000.001;
    double val_resolution = 0.5;
    double test_min = -1000.001;
    double test_max = 1000.001;
    double test_resolution = 0.5;

    //Here are the hyperparameters one can play with.
    int milestone = 50;
    double gamma = 1.5;
    int num_epoch = 500;
    double learning_rate = 0.15;
    int batch_size = 16;
    int val_frequency = 1;

    //The target function and the training data is defined through the Preprocess class and its members.
    //The setProcess() member function awaits the target function, the bounds of the trainin interval and a resolution to sample it.
    Preprocess<double> preprocess;
    preprocess.setProcess([](double x){return x*x;}, train_min, train_max, train_resolution);

    //The scaled dataset is avaliable through the data_scaled() and target_scaled() member functions.
    //They are put here into a Dataset struct.

    Preprocess<double>::Dataset train_dataset;
    train_dataset.data = preprocess.data_scaled(); train_dataset.target = preprocess.target_scaled();

    //Here one can add Gaussian noise. The bool values determine whehter the noise is apllied.
    //To the input data.
    bool add_data_noise = 0;
    double mu_data = 0.;
    double sigma_data = 0.005;
    //To the target.
    bool add_target_noise = 1;
    double mu_target = 0.;
    double sigma_target = 0.0004;

    //The validation and the test dataset can also be generated with the help of the same Preprocess object.
    //The data is generated similarly to the train data with the generate_points() member function. The output is the Dataset struct.
    Preprocess<double>::Dataset val_dataset = preprocess.generate_points(val_min, val_max, val_resolution);

    Preprocess<double>::Dataset test_dataset = preprocess.generate_points(test_min, test_max, test_resolution);

    //The training itself happens here. The train() function requires the train and validation data, the architecture and the hyperparameters to run.
    //The get_bounds() member function is used to carry the scaling parameters for the validation.
    //The validation data is accessible from their respective Dataset struct through the data and target elements.
    //At the end one can specify the addition of random noise to the train data and the train target separately.
    train_out<double> train_output =  train<double>(train_dataset.data, train_dataset.target, num_epoch, learning_rate, batch_size, nn_architecture, 
                                                    val_dataset.data, val_dataset.target, preprocess.get_bounds(), gamma, milestone, val_frequency, 
                                                    add_data_noise, add_target_noise, mu_data, sigma_data, mu_target, sigma_target);

    //The loss values are written out in a text file here.
    cost_history_to_file("cost_square_noisy_target.txt", train_output.cost_history, train_output.val_cost_history, val_frequency);

    //Here happens the prediction on the test dataset. The test dataset is accessible in the same way as the validation dataset.
    std::vector<std::vector<double>> predictions = predict<double>(test_dataset.data, train_output.network_params, nn_architecture, preprocess.get_bounds());

    //Finally, the test data points, ground truth and prediction values are written out into another text file.
    results_to_file<double>("results_square_noisy_target.txt", test_dataset.data, test_dataset.target, predictions);

    return 0;
}
