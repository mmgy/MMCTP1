#include "nn.h"

int main(int, char**) {

    config layer1{1, 16, "sigmoid"};
    config layer2{16, 64, "relu"};
    config layer3{64, 1, "relu"};

    std::vector<config> nn_architecture{layer1, layer2, layer3};

    Preprocess<double> preprocess;
    preprocess.setProcess([](double x){return x*x;}, -50., 50., 1.);

    int milestone = 250;
    double gamma = 1.5;
    int num_epoch = 2500;
    double learning_rate = 0.1;
    int batch_size = 10;
    train_out<double> train_output =  train<double>(preprocess.data_scaled(), preprocess.target_scaled(), num_epoch, learning_rate, batch_size, nn_architecture, gamma, milestone);

    cost_history_to_file("cost_square_try.txt", train_output.cost_history);

    Preprocess<double>::Dataset test_dataset = preprocess.generate_points(0., 100., 1.);

    std::vector<std::vector<double>> predictions = predict<double>(test_dataset.data, train_output.network_params, nn_architecture, preprocess.get_bounds());

    results_to_file<double>("results_square_try.txt", test_dataset.data, test_dataset.target, predictions);
    
    /*
    int k = 5;
    std::vector<double> prediction = point_predict(preprocess.dataset.data[k], train_output.network_params, nn_architecture, preprocess.get_bounds());
    std::vector<double> GT{preprocess.dataset.target[k]};
    std::cout << preprocess.dataset.data[k][0] << " " << preprocess.dataset.target[k] << " " << prediction[0] << " " << RMSE(prediction, GT) << std::endl;*/


    return 0;
}
