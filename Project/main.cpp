#include "nn.h"

int main(int, char**) {
    /*
    std::vector<std::vector<double>> train_data;
    std::vector<double> train_target;

    for(int i = 0; i < 101; i++)
    {
        std::vector<double> train_point{i - 50.};
        train_data.push_back(train_point);
        train_target.push_back((i - 50.) * (i - 50.));
    }

    train_data = MinMaxScaler(train_data, -50., 50.);

    train_target = MinMaxScaler(train_target, 0., 2500.);

    std::shuffle(train_data.begin(), train_data.end(), std::default_random_engine());
    std::shuffle(train_target.begin(), train_target.end(), std::default_random_engine());*/

    config layer1{1, 16, "sigmoid"};
    config layer2{16, 64, "relu"};
    config layer3{64, 1, "relu"};

    std::vector<config> nn_architecture{layer1, layer2, layer3};

    Preprocess<double> preprocess;
    preprocess.setProcess([](double x){return sin(x);}, -5., 5., 0.05);

    int milestone = 250;
    double gamma = 1.5;
    int num_epoch = 2500;
    double learning_rate = 0.1;
    int batch_size = 10;
    train_out<double> train_output =  train<double>(preprocess.data_scaled(), preprocess.target_scaled(), num_epoch, learning_rate, batch_size, nn_architecture, gamma, milestone);

    cost_history_to_file("cost_sin_try.txt", train_output.cost_history);

    //train_data = invMinMaxScaler(train_data, -50., 50.);
    //train_target = invMinMaxScaler(train_target, 0., 2500.);

    std::vector<std::vector<double>> predictions = predict<double>(preprocess.dataset.train_data, train_output.network_params, nn_architecture, preprocess.get_bounds());

    //predictions = invMinMaxScaler(predictions, 0., 2500.);

    results_to_file<double>("results_sin_try.txt", preprocess.dataset.train_data, preprocess.dataset.train_target, predictions);
    
    int k = 5;
    std::vector<double> prediction = point_predict(preprocess.dataset.train_data[k], train_output.network_params, nn_architecture, preprocess.get_bounds());
    std::vector<double> GT{preprocess.dataset.train_target[k]};
    std::cout << preprocess.dataset.train_data[k][0] << " " << preprocess.dataset.train_target[k] << " " << prediction[0] << " " << RMSE(prediction, GT) << std::endl;


    return 0;
}
