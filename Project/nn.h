//Header file for my project: Neural network as a generic function approximator.
//This file includes all the functions and classes that create the pipeline built on std::vectors.

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>
#include <fstream>

#ifndef NN_H
#define NN_H

//Constant to prevent zero division. See later at the cost function.
#define EPS 1e-8
//Config class used to define a fully connected layer by giving the necessary parameters.
typedef struct config
{ 
    int input_dim, output_dim;
    std::string activation;
}config;

//Class to manage the weights associated to the linear part of layers
template<typename T>
class layer_weights
{   
    int size_in, size_out;
    
    void setW()
    {
        W = std::vector<T>(size_out * size_in);
    }

    void setb()
    {
        b = std::vector<T>(size_out);
    }
    
    public:
        std::vector<T> W;
        std::vector<T> b;

        void setSize(int const& set_size_in, int const& set_size_out)
        {
            size_in = set_size_in;
            size_out = set_size_out;
            setW();
            setb();            
        }
};
//Class inherited from layer_weights containing an extra vector which allows the gradient to flow through the layer 
template<typename T>
class backprop_out: public layer_weights<T>
{
    void set_prev()
    {
        prev = std::vector<T>(size_in);
    }

    public:
        std::vector<T> prev;

        void setSize(int const& set_size_in, int const& set_size_out)
        {
            layer_weights<T>::setSize(set_size_in, set_size_out);
            set_prev();
        }
};
//Class to manage the outputs of a single epoch of training
template<typename T>
class train_step_out
{
    public:
        T cost;
        std::vector<layer_weights<T>> network_params;
};
//Class to manage the outputs of the full training process
template<typename T>
class train_out
{
    public:
        std::vector<T> cost_history;
        std::vector<T> val_cost_history;
        std::vector<layer_weights<T>> network_params;
};
//Class to save the weights and outputs through a full forward step. These will be feeded into the backpropagation.
template<typename T>
class nn_out
{
    int size, num_layers;

    void set_size()
    {
        std::vector<T> output = std::vector<T>(size);
    }

    void set_num_layers()
    {
        std::vector<std::vector<std::vector<T>>> forward = std::vector<std::vector<std::vector<T>>>(num_layers);
    }

    public:
        std::vector<T> output;
        std::vector<std::vector<std::vector<T>>> forward;

        void setSize(int const& set_size, int const& set_num_layers)
        {
            size = set_size;
            num_layers = set_num_layers;
            set_size();
            set_num_layers();
        }
};
//Class to create, scale the dataset with the desired target function. Also can generate validation and test datasets.
template<typename T>
class Preprocess
{
    T low, high, resolution, min, max, scaled_data_min, scaled_data_max, scaled_target_min, scaled_target_max;
    std::function<T(T const&)> function;

    public:
        struct Dataset
        {
            std::vector<std::vector<T>> data;
            std::vector<T> target;
        };

        Dataset dataset;

        void setProcess(std::function<T(T const&)> set_function, T const& set_low, T const& set_high, T const& set_resolution)
        {
            low = set_low;
            high = set_high;
            resolution = set_resolution;
            function = set_function;
            Dataset set_dataset = generate_points(low, high, resolution);
            auto result = std::minmax_element(set_dataset.target.begin(), set_dataset.target.end());
            min = *result.first;
            max = *result.second;
            std::shuffle(set_dataset.data.begin(), set_dataset.data.end(), std::default_random_engine());
            std::shuffle(set_dataset.target.begin(), set_dataset.target.end(), std::default_random_engine());
            dataset = set_dataset;
            scaled_data_min = low;
            scaled_data_max = high;
            scaled_target_min = min;
            scaled_target_max = max;
        }

        std::vector<std::vector<T>> data_scaled(T const& set_scaled_data_min = 0, T const& set_scaled_data_max = 1)
        {   
            scaled_data_min = set_scaled_data_min;
            scaled_data_max = set_scaled_data_max;
            return MinMaxScaler(dataset.data, low, high, scaled_data_min, scaled_data_max);
        }

        std::vector<T> target_scaled(T const& set_scaled_target_min = 0, T const& set_scaled_target_max = 1)
        {
            scaled_target_min = set_scaled_target_min;
            scaled_target_max = set_scaled_target_max;
            return MinMaxScaler(dataset.target, min, max, scaled_target_min, scaled_target_max);
        }

        std::vector<std::vector<T>> data()
        {
            return dataset.data;
        }

        std::vector<T> target()
        {
            return dataset.target;
        }

        std::vector<T> get_bounds()
        {
            std::vector<T> bounds;
            bounds.push_back(low);
            bounds.push_back(high);
            bounds.push_back(min);
            bounds.push_back(max);
            bounds.push_back(scaled_data_min);
            bounds.push_back(scaled_data_max);
            bounds.push_back(scaled_target_min);
            bounds.push_back(scaled_target_max);
            return bounds;
        }

        Dataset generate_points(T const& set_low, T const& set_high, T const& set_resolution)
        {
            std::vector<std::vector<T>> generated_data;
            std::vector<T> generated_target;
            Dataset generated_dataset;
            int n = (int)((set_high - set_low) / set_resolution);
            if(n * set_resolution != set_high - set_low)
            {
                n += 1;
            }
            for(int i = 1; i <= n; i++)
            {   
                if(i == n)
                {
                    std::vector<T> train_point{set_high};
                    generated_target.push_back(function(train_point[0]));
                    generated_data.push_back(train_point);
                }else
                {
                    std::vector<T> train_point{set_low + i * set_resolution};
                    generated_target.push_back(function(train_point[0]));
                    generated_data.push_back(train_point);
                }
            }
            generated_dataset.data = generated_data;
            generated_dataset.target = generated_target;
            return generated_dataset;
        }
};
//Initialize the weights of each layers randomly with real values ranging form -1. to 1.
template<typename T>
std::vector<layer_weights<T>> init_network(std::vector<config> const& nn_architecture)
{
    int number_of_layers = static_cast<int>(nn_architecture.size());
    std::vector<layer_weights<T>> network_params;

    for(int i = 0; i < number_of_layers; i++)
    {
        int layer_input_size = nn_architecture[i].input_dim;
        int layer_output_size = nn_architecture[i].output_dim;
        
        layer_weights<T> layer;
        layer.setSize(layer_input_size, layer_output_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(42 + i);
        std::uniform_real_distribution<T> dist(-1., 1.);

        std::generate(layer.W.begin(), layer.W.end(), [&]{return dist(gen);});
        std::generate(layer.b.begin(), layer.b.end(), [&]{return dist(gen);});

        network_params.push_back(layer);
    }
    return network_params;
}
//The standard RelU activation function. Here is where the non-lineariy enters.
template<typename T>
std::vector<T> ReLU(std::vector<T> const& x)
{
    std::vector<T> y(x);
    for(int i = 0; i < static_cast<int>(y.size()); i++)
    {
        if(y[i] < 0.){
            y[i] = 0.;
        }
    }
    return y;
}
//Backward ReLU to use in backpropagation.
template<typename T>
std::vector<T> ReLU_back(std::vector<T> const& dA, std::vector<T> const& x)
{
    std::vector<T> dx(dA);
    for(int i = 0; i < static_cast<int>(x.size()); i++)
    {
        if(x[i] <= 0.){
            dx[i] = 0.;
        }
    }
    return dx;
}
//Sigmoid funtion, another widely known non-linear activation function.
template<typename T>
std::vector<T> sigmoid(std::vector<T> const& x)
{
    std::vector<T> y(x);
    for(int i = 0; i < static_cast<int>(y.size()); i++)
    {
        y[i] = 1. / (1. + exp(-y[i]));
    }
    return y;
}
//Backward sigmoid to use in backpropagation.
template<typename T>
std::vector<T> sigmoid_back(std::vector<T> const& dA, std::vector<T> const& x)
{
    std::vector<T> dx(dA);
    for(int i = 0; i < static_cast<int>(dx.size()); i++)
    {
        dx[i] *= sigmoid(x)[i] * (1 - sigmoid(x)[i]);
    }
    return dx;
}
//Performs the forward propagation for an input datapoint.
template<typename T>
nn_out<T> forward_propagation(std::vector<T> const& input, std::vector<layer_weights<T>> const& network_params, std::vector<config> const& nn_architecture)
{
    std::vector<std::vector<std::vector<T>>> forward;
    std::vector<T> curr_input(input);
    int number_of_layers = static_cast<int>(nn_architecture.size());
    
    for(int i = 0; i < number_of_layers; i++)
    {
        std::vector<std::vector<T>> forward_element;
        std::vector<T> prev_input(curr_input);

        std::vector<T> lin_output = lin_trf(network_params[i], prev_input);

        if(nn_architecture[i].activation == "relu")
        {
            curr_input = ReLU(lin_output);
        }else if(nn_architecture[i].activation == "sigmoid")
        {
            curr_input = sigmoid(lin_output);
        }

        forward_element.push_back(prev_input);
        forward_element.push_back(lin_output);
        forward.push_back(forward_element);
    }

    nn_out<T> prop_out;
    prop_out.output = curr_input;
    prop_out.forward = forward;

    return prop_out;
}
//Support function for the vector arithmetics.
template<typename T>
T dotProduct(std::vector<T> const& A, std::vector<T> const& B)
{
    T product = 0.;
    for(int i = 0; i < static_cast<int>(B.size()); i++)
    {
        product += A[i] * B[i];
    }
    return product;
}
//Support function for the vector arithmetics.
template<typename T>
std::vector<T> matrix_dot_vector(std::vector<T> const& M, std::vector<T> const& v)
{
    std::vector<T> result;
    for(int i = 0; i < (static_cast<int>(M.size()) / static_cast<int>(v.size())); i++)
    {
        std::vector<T> w = std::vector<T>(M.begin() + i*static_cast<int>(v.size()), M.begin() + (i+1)*static_cast<int>(v.size()));
        result.push_back(dotProduct(w, v));
    }
    return result;
}
//Support function for the vector arithmetics.
template<typename T>
std::vector<T> diadicProduct(std::vector<T> const& A, std::vector<T> const& B)
{
    std::vector<T> M(static_cast<int>(A.size()) * static_cast<int>(B.size()));
    for(int i = 0; i < static_cast<int>(A.size()); i++)
    {
        for(int j = 0; j < static_cast<int>(B.size()); j++)
        {
            M[i * static_cast<int>(B.size()) + j] = A[i] * B[j];
        }
    }
    return M;
}
//Support function for the vector arithmetics.
template<typename T>
std::vector<T> transpose(std::vector<T> const& M, int const& row_size)
{
    int num_rows = static_cast<int>(M.size()) / row_size;
    std::vector<T> M_T(num_rows * row_size);
    for(int i = 0; i < row_size; i++)
    {
        for(int j = 0; j < num_rows; j++)
        {
            M_T[i * num_rows + j] = M[j * row_size + i];
        }
    }
    return M_T;
}
//Performs the linear transformation of a layer.
template<typename T>
std::vector<T> lin_trf(layer_weights<T> const& layer, std::vector<T> const& x)
{
    std::vector<T> result;
    for(int i = 0; i < (static_cast<int>(layer.W.size()) / static_cast<int>(x.size())); i++)
    {
        std::vector<T> w = std::vector<T>(layer.W.begin() + i*static_cast<int>(x.size()), layer.W.begin() + (i+1)*static_cast<int>(x.size()));
        result.push_back(dotProduct(w, x) + layer.b[i]);
    }
    return result;
}
//Loss funtion to minimise during training.
template<typename T>
T RMSE(std::vector<T> const& y_hat, std::vector<T> const& y)
{
    T result = 0;
    int n = static_cast<int>(y.size());
    for(int i = 0; i < n; i++)
    {
        result += (y_hat[i] - y[i]) * (y_hat[i] - y[i]) / n;
    }
    return sqrt(result);
}
//Derivative of the loss function. The EPS parameter is needed here in case the prediction is the exact same as the ground truth.
template<typename T>
std::vector<T> dRMSE(std::vector<T> const& y_hat, std::vector<T> const& y)
{
    std::vector<T> result;
    int n = static_cast<int>(y.size());
    T rmse = RMSE(y_hat, y);
    for(int i = 0; i < n; i++)
    {
        result.push_back((y_hat[i] - y[i]) / n / (rmse + EPS));
    }
    return result;
}
//Performs the backpropagation of the gradients through one layer.
template<typename T>
backprop_out<T> single_layer_backprop(std::vector<T> const& d_curr, std::vector<T> const& W_curr, std::vector<T> const& b_curr, std::vector<T> const& lin_output, 
                                      std::vector<T> const& prev_input, std::function<std::vector<T>(std::vector<T> const&, std::vector<T> const&)> backward_activation)
{
    std::vector<T> d_lin_output = backward_activation(d_curr, lin_output);

    std::vector<T> dW_curr = diadicProduct(d_lin_output, prev_input);
    std::vector<T> db_curr = d_lin_output;
    std::vector<T> d_prev = matrix_dot_vector(transpose(W_curr, static_cast<int>(W_curr.size()) / static_cast<int>(d_lin_output.size())), d_lin_output);

    backprop_out<T> single_layer_backprop_out;
    single_layer_backprop_out.W = dW_curr;
    single_layer_backprop_out.b = db_curr;
    single_layer_backprop_out.prev = d_prev;

    return single_layer_backprop_out;
}
//Performs the backpropagation through the whole network.
template<typename T>
std::vector<layer_weights<T>> full_backprop(std::vector<T> const& y_hat, std::vector<T> const& y, std::vector<std::vector<std::vector<T>>> const& forward, 
                                            std::vector<layer_weights<T>> const& network_params, std::vector<config> const& nn_architecture)
{
    std::vector<layer_weights<T>> grads;
    std::vector<T> d_prev = dRMSE(y_hat, y);
    
    int n = static_cast<int>(nn_architecture.size());
    for(int i = 0; i < n; i++)
    {
        layer_weights<T> layer;
        int layer_input_size = nn_architecture[n - i - 1].input_dim;
        int layer_output_size = nn_architecture[n - i - 1].output_dim;
        layer.setSize(layer_input_size, layer_output_size);

        std::function<std::vector<T>(std::vector<T> const&, std::vector<T> const&)> backward_activation;
        if(nn_architecture[i].activation == "relu")
        {
            backward_activation = ReLU_back<T>;
        }else if(nn_architecture[i].activation == "sigmoid")
        {
            backward_activation = sigmoid_back<T>;
        }
        std::vector<T> d_curr = d_prev;
        
        std::vector<T> prev_input = forward[n - i - 1][0];
        std::vector<T> lin_output = forward[n - i - 1][1];
        std::vector<T> W_curr = network_params[n - i - 1].W;
        std::vector<T> b_curr = network_params[n - i - 1].b;
        
        backprop_out<T> single_layer_backprop_out = single_layer_backprop(d_curr, W_curr, b_curr, lin_output, prev_input, backward_activation);
        layer.W = single_layer_backprop_out.W;
        layer.b = single_layer_backprop_out.b;
        d_prev = single_layer_backprop_out.prev;
        
        grads.push_back(layer);
    }
    return grads;
}
//Updates the weights of the network weights with the calculated gradients.
template<typename T>
std::vector<layer_weights<T>> update(std::vector<layer_weights<T>>& network_params, std::vector<layer_weights<T>> const& grads, std::vector<config> const& nn_architecture, T const& learning_rate)
{
    int n = static_cast<int>(nn_architecture.size());
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < static_cast<int>(network_params[i].W.size()); j++)
        {
            network_params[i].W[j] -= learning_rate * grads[n - i - 1].W[j];
        }
        for(int k = 0; k < static_cast<int>(network_params[i].b.size()); k++)
        {
            network_params[i].b[k] -= learning_rate * grads[n - i - 1].b[k];
        }
    }
    return network_params;
}
//Averages the gradients within one batch.
template<typename T>
std::vector<layer_weights<T>> batch_average(std::vector<std::vector<layer_weights<T>>> const& grads_batch)
{
    std::vector<layer_weights<T>> average_batch_grads;
    for(int i = 0; i < static_cast<int>(grads_batch[0].size()); i++)
    {
        layer_weights<T> batch_layer;
        batch_layer.setSize(static_cast<int>(grads_batch[0][i].W.size()), static_cast<int>(grads_batch[0][i].b.size()));
        for(int j = 0; j < static_cast<int>(grads_batch[0][i].W.size()); j++)
        {
            batch_layer.W[j] = 0;
            for(int k = 0; k < static_cast<int>(grads_batch.size()); k++)
            {
                batch_layer.W[j] += grads_batch[k][i].W[j] / static_cast<int>(grads_batch.size());
            }
        }
        for(int l = 0; l < static_cast<int>(grads_batch[0][i].b.size()); l++)
        {
            batch_layer.b[l] = 0;
            for(int m = 0; m < static_cast<int>(grads_batch.size()); m++)
            {
                batch_layer.b[l] += grads_batch[m][i].b[l] / static_cast<int>(grads_batch.size());
            }
        }
        average_batch_grads.push_back(batch_layer);
    }
    return average_batch_grads;
}
//Performs the training step of an epoch for a batch of datapoints
template<typename T>
train_step_out<T> train_step(std::vector<std::vector<T>> const& Xs, std::vector<std::vector<T>> const& Ys, std::vector<layer_weights<T>>& network_params, T const& learning_rate, std::vector<config> const& nn_architecture)
{
    train_step_out<T> train_step_output;
    T costs = 0;
    std::vector<std::vector<layer_weights<T>>> grads_batch;
   
    for(int i = 0; i < static_cast<int>(Xs.size()); i++)
    {
         nn_out<T> forward_out = forward_propagation(Xs[i], network_params, nn_architecture);
        T cost = RMSE(forward_out.output, Ys[i]);
        costs += cost / static_cast<int>(Xs.size());
        

        std::vector<layer_weights<T>> grads = full_backprop(forward_out.output, Ys[i], forward_out.forward, network_params, nn_architecture);
        grads_batch.push_back(grads);
    }
    std::vector<layer_weights<T>> average_batch_grads = batch_average(grads_batch);
    network_params = update(network_params, average_batch_grads, nn_architecture, learning_rate);
    train_step_output.cost = costs;
    train_step_output.network_params = network_params;

    return train_step_output;
}
//Generates point prediction from the trained network.
template<typename T>
std::vector<T> point_predict(std::vector<T> const& X, std::vector<layer_weights<T>>& network_params, std::vector<config> const& nn_architecture, 
                             std::vector<T> const& bounds)
{
    T low = bounds[0]; T high = bounds[1]; T min = bounds[2]; T max = bounds[3];
    T scaled_data_min = bounds[4]; T scaled_data_max = bounds[5]; T scaled_target_min = bounds[6]; T scaled_target_max = bounds[7];
    std::vector<T> X_scaled = MinMaxScaler(X, low, high, scaled_data_min, scaled_data_max);
    nn_out<T> forward_out = forward_propagation(X_scaled, network_params, nn_architecture);
    return invMinMaxScaler(forward_out.output, min, max, scaled_target_min, scaled_target_max);
}
//Generates predictions for a vector of points from the trained network.
template<typename T>
std::vector<std::vector<T>> predict(std::vector<std::vector<T>> const& data, std::vector<layer_weights<T>>& network_params, std::vector<config> const& nn_architecture, 
                                    std::vector<T> const& bounds)
{
    std::vector<std::vector<T>> predictions;
    for(int i = 0; i < static_cast<int>(data.size()); i++)
    {
        predictions.push_back(point_predict(data[i], network_params, nn_architecture, bounds));
    }
    return predictions;
}
//Writes the data points, their ground truth images and the predictions into a file.
template<typename T>
void results_to_file(std::string file_name, std::vector<std::vector<T>> const& data, std::vector<T> const& target, std::vector<std::vector<T>> predictions)
{
    std::ofstream output(file_name);
    if(output.is_open())
    {
        std::vector<std::string> data_header = {"Data point", "Image", "Prediction"};

        std::copy(data_header.begin(), data_header.end(), std::ostream_iterator<std::string>(output, " "));

        output << "\n";

        for(int i = 0; i < static_cast<int>(data.size()); i++)
        {
            output << data[i][0] << ", " << target[i] << ", " << predictions[i][0] << "\n" << std::endl; 
        }
    }else
    {
        std::cout << "Could not open output file\n" << std::endl;
    }
}
//Writes losses for the predictions for a dataset into a file.
template<typename T>
void cost_history_to_file(std::string const& file_name, std::vector<T> const& cost_history, std::vector<T> const& val_cost_history, int const& val_frequency)
{
    std::ofstream output(file_name);
    if(output.is_open())
    {
        std::vector<std::string> data_header = {"Epoch", "RMSE loss", "Validation RMSE loss"};

        std::copy(data_header.begin(), data_header.end(), std::ostream_iterator<std::string>(output, " "));

        output << "\n";

        for(int i = 0; i < static_cast<int>(cost_history.size()); i++)
        {
            if((i + 1) % val_frequency == 0)
            {
                output << i + 1 << ", " << cost_history[i] << ", " << val_cost_history[i] <<  "\n" << std::endl;
            }else
            {
                output << i + 1 << ", " << cost_history[i] << "\n" << std::endl;
            } 
        }
    }else
    {
        std::cout << "Could not open output file\n" << std::endl;
    }
}
//Perform the whole training process on a vector of datapoints and their target values.
template<typename T>
train_out<T> train(std::vector<std::vector<T>>& train_data, std::vector<T>& train_target, int const& num_epoch, T const& learning_rate, 
                   int const& batch_size, std::vector<config> const& nn_architecture, std::vector<std::vector<T>> const& val_data, std::vector<T> val_target, 
                   std::vector<T> const& bounds, double const& gamma = 1.5, int const& milestone = 1000, int const& val_frequency = 10, 
                   bool add_data_noise = 0, bool add_target_noise = 0, T const& mu_data = 0., T const& sigma_data = 1., T const& mu_target = 0., T const& sigma_target = 1.)
{
    std::vector<layer_weights<double>> network_params = init_network<double>(nn_architecture);
    train_step_out<T> train_step_output;
    std::vector<T> cost_history;
    std::vector<T> val_cost_history;
    std::vector<std::vector<T>> train_target_vectorized;

    for(int p = 0; p < static_cast<int>(train_target.size()); p++)
    {
        train_target_vectorized.push_back(std::vector<T>{train_target[p]});
    }
    for(int i = 0; i < num_epoch; i++)
    {
        if(add_data_noise)
        {
            train_data = addGaussianNoise(train_data, mu_data, sigma_data, i);
        }
        if(add_target_noise)
        {
            train_target = addGaussianNoise(train_target, mu_target, sigma_target, i);
        }

        T cost_epoch_mean = 0;
        int m = static_cast<int>(train_data.size()) / batch_size;
        if(m * batch_size != static_cast<int>(train_data.size()))
        {
            m += 1;
        }
        for(int j = 0; j < m; j++)
        {
            std::vector<std::vector<T>> Xs;
            std::vector<std::vector<T>> Ys;
            if(j == m - 1)
            {
                Xs = std::vector<std::vector<T>>(train_data.begin() + j * batch_size, train_data.end());
                Ys = std::vector<std::vector<T>>(train_target_vectorized.begin() + j * batch_size, train_target_vectorized.end());
            }else
            {
                Xs = std::vector<std::vector<T>>(train_data.begin() + j * batch_size, train_data.begin() + (j + 1) * batch_size);
                Ys = std::vector<std::vector<T>>(train_target_vectorized.begin() + j * batch_size, train_target_vectorized.begin() + (j + 1) * batch_size);
            }
            train_step_output = train_step(Xs, Ys, network_params, learning_rate * pow(gamma, -i / milestone), nn_architecture);
            if(j == 0)
            {
                cost_epoch_mean += train_step_output.cost;
            }else
            {
                cost_epoch_mean *= 1. / cost_epoch_mean * (j * cost_epoch_mean + train_step_output.cost) / (1 + j);
            }
        }
        cost_history.push_back(cost_epoch_mean);

        if((i + 1) % val_frequency == 0)
        {
            std::vector<std::vector<T>> predictions = predict<T>(val_data, train_step_output.network_params, nn_architecture, bounds);
            T val_loss = 0.;
            for(int q = 0; q < static_cast<int>(predictions.size()); q++)
            {
                std::vector<T> val_prediction_value = MinMaxScaler(predictions[q], bounds[2], bounds[3], bounds[6], bounds[7]);
                std::vector<T> val_target_value{val_target[q]};
                val_target_value = MinMaxScaler(val_target_value, bounds[2], bounds[3], bounds[6], bounds[7]);
                val_loss += RMSE(val_prediction_value, val_target_value) / static_cast<int>(predictions.size());
            }
            std::cout << "learning rate: " << learning_rate * pow(gamma, -i / milestone) << std::endl;
            std::cout << "RMSE loss after " << (i + 1) << " epochs: " << cost_epoch_mean << ", val_loss: " << val_loss << "\n" <<std::endl;
            val_cost_history.push_back(val_loss);
        }else
        {
            std::cout << "learning rate: " << learning_rate * pow(gamma, -i / milestone) << std::endl;
            std::cout << "RMSE loss after " << (i + 1) << " epochs: " << cost_epoch_mean << "\n" <<std::endl;
        }
    }

    train_out<T> train_output;
    train_output.cost_history = cost_history;
    train_output.val_cost_history = val_cost_history;
    train_output.network_params = network_params;

    return train_output;
}
//Scales the data points to a given range of values according to their minimum and maximum values.
template<typename T>
std::vector<std::vector<T>> MinMaxScaler(std::vector<std::vector<T>> const& data, T const& data_min, T const& data_max, T min = 0., T max = 1.)
{
    std::vector<std::vector<T>> data_scaled;
    for(int i = 0; i < static_cast<int>(data.size()); i++)
    {
        std::vector<T> data_scaled_element{((data[i][0] - data_min) / (data_max - data_min)) * (max - min) + min};
        data_scaled.push_back(data_scaled_element);
    }
    return data_scaled;
}
//Inverse scaling for both cases.
template<typename T, typename F>
F invMinMaxScaler(F const& data_scaled, T const& data_min, T const& data_max, T min = 0., T max = 1.)
{
    F data = MinMaxScaler(data_scaled, min, max, data_min, data_max);
    return data;
}
//Scales the targets to a given range of values according to their minimum and maximum values.
template<typename T>
std::vector<T> MinMaxScaler(std::vector<T> const& data, T const& data_min, T const& data_max, T min = 0., T max = 1.)
{
    std::vector<T> data_scaled;
    for(int i = 0; i < static_cast<int>(data.size()); i++)
    {
        T data_scaled_element{((data[i] - data_min) / (data_max - data_min)) * (max - min) + min};
        data_scaled.push_back(data_scaled_element);
    }
    return data_scaled;
}

template<typename T>
std::vector<std::vector<T>> addGaussianNoise(std::vector<std::vector<T>>& data, T const& mu = 0., T const& sigma = 1., int const& seed = 0)
{
    std::default_random_engine generator;
    generator.seed(seed);
    std::normal_distribution<T> distribution(mu, sigma);

    for(int i = 0; i < static_cast<int>(data.size()); i++)
    {
        data[i][0] += distribution(generator);
    }
    return data;
}

template<typename T>
std::vector<T> addGaussianNoise(std::vector<T>& target, T const& mu = 0., T const& sigma = 1., int const& seed = 0)
{
    std::default_random_engine generator;
    generator.seed(seed);
    std::normal_distribution<T> distribution(mu, sigma);

    for(int i = 0; i < static_cast<int>(target.size()); i++)
    {
        target[i] += distribution(generator);
    }
    return target;
}

#endif