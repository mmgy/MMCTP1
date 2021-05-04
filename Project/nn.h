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

#define EPS 1e-8

typedef struct config
{ 
    int input_dim, output_dim;
    std::string activation;
}config;

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

template<typename T>
class train_step_out
{
    public:
        T cost;
        std::vector<layer_weights<T>> network_params;
};

template<typename T>
class train_out
{
    public:
        std::vector<T> cost_history;
        std::vector<layer_weights<T>> network_params;
};


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
        //std::cout << lin_output[0] << std::endl;

        forward_element.push_back(prev_input);
        forward_element.push_back(lin_output);
        forward.push_back(forward_element);
    }

    //std::cout << "\n End of layers \n" << std::endl;
    nn_out<T> prop_out;
    prop_out.output = curr_input;
    //std::cout << curr_input[0] << "\n" << std::endl;
    prop_out.forward = forward;

    return prop_out;
}

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

template<typename T>
std::vector<T> lin_trf(layer_weights<T> const& layer, std::vector<T> const& x)
{
    /*for(int l = 0; l < x.size(); l++)
    {
        std::cout << x[l] << std::endl;
    }
    std::cout << "\n" << std::endl;*/
    //std::cout << layer.W[0] << std::endl;
    std::vector<T> result;
    for(int i = 0; i < (static_cast<int>(layer.W.size()) / static_cast<int>(x.size())); i++)
    {
        std::vector<T> w = std::vector<T>(layer.W.begin() + i*static_cast<int>(x.size()), layer.W.begin() + (i+1)*static_cast<int>(x.size()));
        result.push_back(dotProduct(w, x) + layer.b[i]);
        //std::cout << dotProduct(w, x) << std::endl;
    }
    return result;
}

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

template<typename T>
std::vector<T> dRMSE(std::vector<T> const& y_hat, std::vector<T> const& y)
{
    std::vector<T> result;
    int n = static_cast<int>(y.size());
    T rmse = RMSE(y_hat, y);
    //std::cout << rmse << std::endl;
    for(int i = 0; i < n; i++)
    {
        result.push_back((y_hat[i] - y[i]) / n / (rmse + EPS));
        //std::cout << (y_hat[i] - y[i]) / n << std::endl;
    }
    return result;
}

template<typename T>
backprop_out<T> single_layer_backprop(std::vector<T> const& d_curr, std::vector<T> const& W_curr, std::vector<T> const& b_curr, std::vector<T> const& lin_output, 
                                      std::vector<T> const& prev_input, std::function<std::vector<T>(std::vector<T> const&, std::vector<T> const&)> backward_activation)
{
    //std::cout << static_cast<int>(d_curr.size()) << std::endl;
    std::vector<T> d_lin_output = backward_activation(d_curr, lin_output);
    //std::cout << static_cast<int>(d_lin_output.size()) << std::endl;

    std::vector<T> dW_curr = diadicProduct(d_lin_output, prev_input);
    std::vector<T> db_curr = d_lin_output;
    //std::cout << static_cast<int>(W_curr.size()) / static_cast<int>(d_lin_output.size()) << std::endl;
    //std::cout << static_cast<int>(transpose(W_curr, static_cast<int>(W_curr.size()) / static_cast<int>(d_lin_output.size())).size()) << std::endl;
    std::vector<T> d_prev = matrix_dot_vector(transpose(W_curr, static_cast<int>(W_curr.size()) / static_cast<int>(d_lin_output.size())), d_lin_output);

    backprop_out<T> single_layer_backprop_out;
    single_layer_backprop_out.W = dW_curr;
    single_layer_backprop_out.b = db_curr;
    single_layer_backprop_out.prev = d_prev;

    return single_layer_backprop_out;
}

template<typename T>
std::vector<layer_weights<T>> full_backprop(std::vector<T> const& y_hat, std::vector<T> const& y, std::vector<std::vector<std::vector<T>>> const& forward, 
                                            std::vector<layer_weights<T>> const& network_params, std::vector<config> const& nn_architecture)
{
    std::vector<layer_weights<T>> grads;
    std::vector<T> d_prev = dRMSE(y_hat, y);
    //std::cout << static_cast<int>(y.size()) << std::endl;
    
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
            //Backward_Activation backward_activation = sigmoid_back;
            backward_activation = sigmoid_back<T>;
        }
        //std::cout << static_cast<int>(d_prev.size()) << std::endl;
        std::vector<T> d_curr = d_prev;
        //std::cout << static_cast<int>(d_curr.size()) << std::endl;
        
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

template<typename T>
std::vector<layer_weights<T>> update(std::vector<layer_weights<T>>& network_params, std::vector<layer_weights<T>> const& grads, std::vector<config> const& nn_architecture, T const& learning_rate)
{
    int n = static_cast<int>(nn_architecture.size());
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < static_cast<int>(network_params[i].W.size()); j++)
        {
            network_params[i].W[j] -= learning_rate * grads[n - i - 1].W[j];
            //std::cout << grads[n - i - 1].W[j] << std::endl;
        }
        for(int k = 0; k < static_cast<int>(network_params[i].b.size()); k++)
        {
            network_params[i].b[k] -= learning_rate * grads[n - i - 1].b[k];
        }
        //std::cout << "\n" << std::endl;
    }
    return network_params;
}

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

template<typename T>
train_step_out<T> train_step(std::vector<std::vector<T>> const& Xs, std::vector<std::vector<T>> const& Ys, std::vector<layer_weights<T>>& network_params, T const& learning_rate, std::vector<config> const& nn_architecture)
{
    train_step_out<T> train_step_output;
    T costs = 0;
    std::vector<std::vector<layer_weights<T>>> grads_batch;
    //std::cout << static_cast<int>(Xs.size()) << std::endl;
   
    for(int i = 0; i < static_cast<int>(Xs.size()); i++)
    {
         nn_out<T> forward_out = forward_propagation(Xs[i], network_params, nn_architecture);
        /*
        for(int l = 0; l < forward_out.output.size(); l++)
        {
            std::cout << forward_out.output[l] << std::endl;
        }
        std::cout << "\n" << std::endl;*/
        //std::cout << forward_out.output[0] << " " << Y[0] << std::endl;
        //std::cout << static_cast<int>(Ys[i].size()) << std::endl;
        T cost = RMSE(forward_out.output, Ys[i]);
        //std::cout << cost << std::endl;
        costs += cost / static_cast<int>(Xs.size());
        

        std::vector<layer_weights<T>> grads = full_backprop(forward_out.output, Ys[i], forward_out.forward, network_params, nn_architecture);
        grads_batch.push_back(grads);
    }
    std::vector<layer_weights<T>> average_batch_grads = batch_average(grads_batch);
    network_params = update(network_params, average_batch_grads, nn_architecture, learning_rate);
    /*for(int l = 0; l < network_params[1].W.size(); l++)
    {
        std::cout << network_params[1].W[l] << std::endl;
    }
    std::cout << "\n" << std::endl;*/
    train_step_output.cost = costs;
    train_step_output.network_params = network_params;
    //std::cout << "\n" << std::endl;

    return train_step_output;
}

template<typename T>
std::vector<T> point_predict(std::vector<T> const& X, std::vector<layer_weights<T>>& network_params, std::vector<config> const& nn_architecture, 
                             std::vector<T> const& bounds)
{
    T low = bounds[0]; T high = bounds[1]; T min = bounds[2]; T max = bounds[3];
    std::vector<T> X_scaled = MinMaxScaler(X, low, high);
    nn_out<T> forward_out = forward_propagation(X_scaled, network_params, nn_architecture);
    return invMinMaxScaler(forward_out.output, min, max);
}

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

template<typename T>
void cost_history_to_file(std::string const& file_name, std::vector<T> const& cost_history)
{
    std::ofstream output(file_name);
    if(output.is_open())
    {
        std::vector<std::string> data_header = {"Epoch", "RMSE loss"};

        std::copy(data_header.begin(), data_header.end(), std::ostream_iterator<std::string>(output, " "));

        output << "\n";

        for(int i = 0; i < static_cast<int>(cost_history.size()); i++)
        {
            output << i + 1 << ", " << cost_history[i] << ", " <<  "\n" << std::endl; 
        }
    }else
    {
        std::cout << "Could not open output file\n" << std::endl;
    }
}

template<typename T>
train_out<T> train(std::vector<std::vector<T>> const& train_data, std::vector<T> const& train_target, int const& num_epoch, T const& learning_rate, 
                   int const& batch_size, std::vector<config> const& nn_architecture, double const& gamma = 1.5, int const& milestone = 1000)
{
    std::vector<layer_weights<double>> network_params = init_network<double>(nn_architecture);
    train_step_out<T> train_step_output;
    std::vector<T> cost_history;
    std::vector<std::vector<T>> train_target_vectorized;

    for(int p = 0; p < static_cast<int>(train_target.size()); p++)
    {
        train_target_vectorized.push_back(std::vector<T>{train_target[p]});
    }
    //std::cout << static_cast<int>(train_target_vectorized[1].size()) << std::endl;
    /*
    for(int l = 0; l < network_params[1].W.size(); l++)
    {
        std::cout << network_params[1].W[l] << std::endl;
    }*/
    for(int i = 0; i < num_epoch; i++)
    {
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
            //std::cout << j + 1 << std::endl;
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
        std::cout << "RMSE loss after " << (i + 1) << " epochs: " << cost_epoch_mean  << "\n" <<std::endl;
        std::cout << "learning rate: " << learning_rate * pow(gamma, -i / milestone) << std::endl;
    }

    train_out<T> train_output;
    train_output.cost_history = cost_history;
    train_output.network_params = network_params;

    return train_output;
}

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

template<typename T, typename F>
F invMinMaxScaler(F const& data_scaled, T const& data_min, T const& data_max, T min = 0., T max = 1.)
{
    F data = MinMaxScaler(data_scaled, min, max, data_min, data_max);
    return data;
}

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
class Preprocess
{
    T low, high, resolution, min, max;
    std::function<T(T const&)> function;

    struct Dataset
    {
        std::vector<std::vector<T>> train_data;
        std::vector<T> train_target;
    };

    public:
        Dataset dataset;

        void setProcess(std::function<T(T const&)> set_function, T const& set_low, T const& set_high, T const& set_resolution)
        {
            low = set_low;
            high = set_high;
            resolution = set_resolution;
            function = set_function;
            /*int n = (int)((high - low) / resolution);
            if(n * resolution != high - low)
            {
                n += 1;
            }
            for(int i = 1; i <= n; i++)
            {   
                if(i == n)
                {
                    std::vector<T> train_point{high};
                    Dataset.train_target.push_back(function(train_point[0]));
                    Dataset.train_data.push_back(train_point);
                }else
                {
                    std::vector<T> train_point{low + i * resolution};
                    Dataset.train_target.push_back(function(train_point[0]));
                    Dataset.train_data.push_back(train_point);
                }
            }*/
            Dataset set_dataset = generate_points(low, high, resolution);
            auto result = std::minmax_element(set_dataset.train_target.begin(), set_dataset.train_target.end());
            min = *result.first;
            max = *result.second;
            std::shuffle(set_dataset.train_data.begin(), set_dataset.train_data.end(), std::default_random_engine());
            std::shuffle(set_dataset.train_target.begin(), set_dataset.train_target.end(), std::default_random_engine());
            dataset = set_dataset;
        }

        std::vector<std::vector<T>> data_scaled()
        {   
            return MinMaxScaler(dataset.train_data, low, high);
        }

        std::vector<T> target_scaled()
        {
            return MinMaxScaler(dataset.train_target, min, max);
        }

        std::vector<std::vector<T>> data()
        {
            return dataset.train_data;
        }

        std::vector<T> target()
        {
            return dataset.train_target;
        }

        std::vector<T> get_bounds()
        {
            std::vector<T> bounds;
            bounds.push_back(low);
            bounds.push_back(high);
            bounds.push_back(min);
            bounds.push_back(max);
            return bounds;
        }

        typename Dataset generate_points(T const& set_low, T const& set_high, T const& set_resolution)
        {
            std::vector<std::vector<T>> data;
            std::vector<T> target;
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
                    target.push_back(function(train_point[0]));
                    data.push_back(train_point);
                }else
                {
                    std::vector<T> train_point{low + i * resolution};
                    target.push_back(function(train_point[0]));
                    data.push_back(train_point);
                }
            }
            generated_dataset.train_data = data;
            generated_dataset.train_target = target;
            return generated_dataset;
        }
};

#endif