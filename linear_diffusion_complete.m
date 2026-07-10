% Linear Diffusion: Building a Diffusion Model from Linear Components
% MATLAB Implementation of Count Bayesie's Linear Diffusion Model
% Complete implementation in a single file following the original blog post
%
% EXECUTION ORDER:
% 1. LinearDiffusion Constructor
% 2. MNIST Data Loading Functions
% 3. Model Training Functions
% 4. Feature Engineering Functions
% 5. Prediction Functions
% 6. Visualization Functions
% 7. Main Function (called at the end)

%% 1. LinearDiffusion Class Implementation (Constructor)

%% LinearDiffusion Class Implementation
function ld = LinearDiffusion()
    % Constructor - returns struct with methods (MATLAB function handle approach)
    ld = struct();
    
    % Properties
    ld.LATENT_SIZE = 12;
    ld.image_size = 28;
    ld.text_encoder = struct();
    ld.image_encoder = struct();
    ld.model = struct();
end

%% 2. MNIST Data Loading Functions

function [train_images, train_labels, test_images, test_labels] = load_mnist_data()
    % Load MNIST data from the 4 binary files in sequential order
    
    % Get current directory to ensure proper file paths
    current_dir = pwd;
    
    % Define file paths in sequential order
    train_images_file = fullfile(current_dir, 'train-images-idx3-ubyte');
    train_labels_file = fullfile(current_dir, 'train-labels-idx1-ubyte');
    test_images_file = fullfile(current_dir, 't10k-images-idx3-ubyte');
    test_labels_file = fullfile(current_dir, 't10k-labels-idx1-ubyte');
    
    % Check if .mat file exists first (faster loading)
    mat_file = fullfile(current_dir, 'mnist.mat');
    if exist(mat_file, 'file')
        fprintf('Loading from cached %s\n', mat_file);
        data = load(mat_file);
        if isfield(data, 'train_images') && isfield(data, 'train_labels') && ...
           isfield(data, 'test_images') && isfield(data, 'test_labels')
            train_images = data.train_images;
            train_labels = data.train_labels;
            test_images = data.test_images;
            test_labels = data.test_labels;
            fprintf('Loaded from cache: %d training + %d test samples\n', ...
                    length(train_labels), length(test_labels));
            return;
        end
    end
    
    % Load from the 4 MNIST binary files in sequential order
    if exist(train_images_file, 'file') && exist(train_labels_file, 'file') && ...
       exist(test_images_file, 'file') && exist(test_labels_file, 'file')
        
        fprintf('Loading MNIST from 4 binary files in sequential order:\n');
        fprintf('1. Reading %s\n', train_images_file);
        fprintf('2. Reading %s\n', train_labels_file);
        
        % Load training data first (files 1-2)
        [train_images, train_labels] = read_mnist_images_labels(train_images_file, train_labels_file);
        
        fprintf('3. Reading %s\n', test_images_file);
        fprintf('4. Reading %s\n', test_labels_file);
        
        % Load test data second (files 3-4)
        [test_images, test_labels] = read_mnist_images_labels(test_images_file, test_labels_file);
        
        % Save for future use
        fprintf('Caching data to %s for faster future loading...\n', mat_file);
        save(mat_file, 'train_images', 'train_labels', 'test_images', 'test_labels');
        
        fprintf('Successfully loaded %d training + %d test samples\n', ...
                length(train_labels), length(test_labels));
        return;
    end
    
    % Error if files not found - only use the 4 required files
    error(['MNIST binary files not found! Please ensure these 4 files are in the current directory:\n' ...
           '1. train-images-idx3-ubyte\n' ...
           '2. train-labels-idx1-ubyte\n' ...
           '3. t10k-images-idx3-ubyte\n' ...
           '4. t10k-labels-idx1-ubyte']);
end

function [images, labels] = read_mnist_images_labels(image_file, label_file)
    % Read MNIST binary format - only for the 4 main files
    
    % Gunzip if needed (optional for compressed files)
    if endsWith(image_file, '.gz')
        gunzip(image_file);
        image_file = image_file(1:end-3);
    end
    if endsWith(label_file, '.gz')
        gunzip(label_file);
        label_file = label_file(1:end-3);
    end
    
    fprintf('Reading images from: %s\n', image_file);
    fprintf('Reading labels from: %s\n', label_file);
    
    % Read images
    fid = fopen(image_file, 'rb', 'ieee-be');
    if fid == -1
        error('Cannot open image file: %s', image_file);
    end
    
    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magic ~= 2051
        error('Invalid magic number in image file: %d (expected 2051)', magic);
    end
    
    num_images = fread(fid, 1, 'int32', 0, 'ieee-be');
    num_rows = fread(fid, 1, 'int32', 0, 'ieee-be');
    num_cols = fread(fid, 1, 'int32', 0, 'ieee-be');
    
    fprintf('Loading %d images of size %dx%d\n', num_images, num_rows, num_cols);
    
    images = fread(fid, num_images * num_rows * num_cols, 'unsigned char');
    fclose(fid);
    
    if length(images) ~= num_images * num_rows * num_cols
        error('Could not read all image data');
    end
    
    images = reshape(images, num_cols, num_rows, num_images);
    images = permute(images, [3, 2, 1]);
    images = double(images) / 255.0;
    
    % Read labels
    fid = fopen(label_file, 'rb', 'ieee-be');
    if fid == -1
        error('Cannot open label file: %s', label_file);
    end
    
    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magic ~= 2049
        error('Invalid magic number in label file: %d (expected 2049)', magic);
    end
    
    num_labels = fread(fid, 1, 'int32', 0, 'ieee-be');
    if num_labels ~= num_images
        warning('Number of labels (%d) does not match number of images (%d)', num_labels, num_images);
    end
    
    labels = fread(fid, num_labels, 'unsigned char');
    fclose(fid);
    
    if length(labels) ~= num_labels
        error('Could not read all label data');
    end
    
    labels = cellstr(string(labels));
    
    fprintf('Successfully loaded %d images and %d labels\n', size(images, 1), length(labels));
end

%% 3. Model Training Functions
function ld = fit_model(ld, labels, images)
    fprintf('Training Linear Diffusion Model...\n');
    
    % 1. Text embedding - One Hot Encoding (as in blog)
    fprintf('Creating text embeddings...\n');
    text_encoder_temp = create_text_encoder(labels);
    ld.text_encoder = text_encoder_temp;
    label_embeddings = encode_text_labels(ld, labels);
    
    % 2. Image encoding/decoding - PCA (as in blog)
    fprintf('Creating image encoder with PCA...\n');
    [image_encoder_temp, X_encode] = create_pca_encoder(ld, images);
    ld.image_encoder = image_encoder_temp;
    
    % 3. Create features with noise and interaction terms
    fprintf('Creating features with interaction terms...\n');
    [X_train, y_noise] = create_features_noise(ld, label_embeddings, X_encode);
    
    % 4. Train linear regression model (as in blog)
    fprintf('Training linear regression denoiser...\n');
    model_temp = train_linear_regression(X_train, y_noise);
    ld.model = model_temp;
    
    fprintf('Training completed!\n\n');
end

function text_encoder = create_text_encoder(labels)
    % Create one-hot encoder exactly as in the blog
    unique_labels = unique(labels);
    num_categories = length(unique_labels);
    
    text_encoder.categories = unique_labels;
    text_encoder.num_categories = num_categories;
end

function label_embeddings = encode_text_labels(ld, labels)
    % One-hot encode text labels (drop first category as in sklearn)
    num_samples = length(labels);
    num_categories = ld.text_encoder.num_categories;
    
    % Create one-hot matrix, drop first category to avoid multicollinearity
    label_embeddings = zeros(num_samples, num_categories - 1);
    
    for i = 1:num_samples
        label_idx = find(strcmp(ld.text_encoder.categories, labels{i}));
        if label_idx > 1  % Skip first category (dropped)
            label_embeddings(i, label_idx - 1) = 1;
        end
    end
end

function [image_encoder, X_encode] = create_pca_encoder(ld, images)
    % Image encoding/decoding - PCA as described in the blog
    
    % Reshape images to vectors (28x28 = 784 features)
    [num_images, height, width] = size(images);
    all_data = reshape(images, num_images, height * width);
    
    % Standardization (StandardScaler equivalent)
    scaler_mean = mean(all_data, 1);
    scaler_std = std(all_data, 0, 1);
    scaler_std(scaler_std == 0) = 1; % Avoid division by zero
    
    standardized_data = (all_data - scaler_mean) ./ scaler_std;
    
    % PCA with n_components = LATENT_SIZE^2 (12x12 = 144)
    num_components = ld.LATENT_SIZE * ld.LATENT_SIZE;
    
    % Ensure we don't request more components than available
    max_components = min(size(standardized_data, 1) - 1, size(standardized_data, 2));
    num_components = min(num_components, max_components);
    
    fprintf('Performing PCA with %d components (max possible: %d)\n', num_components, max_components);
    
    [coeff, score, latent] = pca(standardized_data);
    
    % Store encoder parameters
    image_encoder.scaler_mean = scaler_mean;
    image_encoder.scaler_std = scaler_std;
    image_encoder.pca_components = coeff(:, 1:num_components);
    image_encoder.pca_mean = mean(standardized_data, 1);
    image_encoder.explained_variance = latent(1:num_components);
    image_encoder.num_components = num_components;  % Store actual number used
    
    % Encoded data (latents)
    X_encode = score(:, 1:num_components);
    
    fprintf('PCA encoding: %dx%d -> %dx%d\n', height, width, ld.LATENT_SIZE, ld.LATENT_SIZE);
end

function decoded_data = decode_from_latent(ld, encoded_data)
    % Decode images from PCA latent space back to pixel space
    
    % Inverse PCA transform
    standardized_data = encoded_data * ld.image_encoder.pca_components' + ld.image_encoder.pca_mean;
    
    % Inverse standardization
    image_vectors = standardized_data .* ld.image_encoder.scaler_std + ld.image_encoder.scaler_mean;
    
    % Reshape back to images
    num_images = size(image_vectors, 1);
    decoded_data = reshape(image_vectors, num_images, ld.image_size, ld.image_size);
end

%% 4. Feature Engineering Functions
function [features, noise] = create_features_noise(ld, text_embeddings, image_embeddings, std_noise, seed)
    % Create features with noise and interaction terms exactly as in blog
    if nargin < 4, std_noise = 1.0; end
    if nargin < 5, seed = 1337; end
    
    rng(seed); % Set random seed
    
    num_samples = size(text_embeddings, 1);
    
    % Use actual latent dimension from encoder (not theoretical)
    if isfield(ld.image_encoder, 'num_components')
        latent_dim = ld.image_encoder.num_components;
    else
        latent_dim = ld.LATENT_SIZE * ld.LATENT_SIZE;
    end
    
    % Generate noise from standard normal (as in blog)
    noise = randn(num_samples, latent_dim) * std_noise;
    
    if nargin >= 3 && ~isempty(image_embeddings)
        % Training: add noise to image embeddings
        noised_embeddings = image_embeddings + noise;
    else
        % Generation: use pure noise
        noised_embeddings = noise;
    end
    
    % Create interaction terms (feature engineering for linear model)
    interaction_terms = create_interaction_terms(text_embeddings, noised_embeddings);
    
    % Concatenate features: [noised_embeddings, text_embeddings, interaction_terms]
    features = [noised_embeddings, text_embeddings, interaction_terms];
end

function interactions = create_interaction_terms(text_embeddings, image_embeddings)
    % Create interaction terms exactly as described in the blog
    [num_samples, text_dim] = size(text_embeddings);
    [~, image_dim] = size(image_embeddings);
    
    interactions = zeros(num_samples, text_dim * image_dim);
    
    for i = 1:text_dim
        start_idx = (i-1) * image_dim + 1;
        end_idx = i * image_dim;
        interactions(:, start_idx:end_idx) = image_embeddings .* text_embeddings(:, i);
    end
end

function model = train_linear_regression(X_train, y_noise)
    % Train linear regression model exactly as in the blog
    
    % Linear regression: solve (X'X + λI)β = X'y for numerical stability
    lambda = 1e-6; % Small regularization
    model.coefficients = (X_train' * X_train + lambda * eye(size(X_train, 2))) \ (X_train' * y_noise);
    
    % Calculate training MSE
    y_pred = X_train * model.coefficients;
    model.mse = mean((y_noise - y_pred).^2, 'all');
    
    fprintf('Training MSE: %.6f\n', model.mse);
end

%% 5. Prediction Functions
function generated_images = predict_images(ld, labels, seed)
    % Generate images from text labels (the main prediction method)
    if nargin < 3, seed = 1337; end
    
    % Convert labels to cell array if needed
    if ischar(labels), labels = {labels}; end
    if isnumeric(labels)
        labels = cellfun(@num2str, num2cell(labels), 'UniformOutput', false);
    end
    
    % Encode text labels
    label_embeddings = encode_text_labels(ld, labels);
    
    % Create features with pure noise (no image embeddings)
    [X_test, noise_test] = create_features_noise(ld, label_embeddings, [], 1.0, seed);
    
    % Predict noise using trained linear model
    est_noise = X_test * ld.model.coefficients;
    
    % Denoise by subtracting estimated noise
    denoised = noise_test - est_noise;
    
    % Decode back to images
    generated_images = decode_from_latent(ld, denoised);
end

%% 6. Visualization Functions (matching blog examples)
function visualize_blog_results(generated_images, labels)
    % Visualize results exactly as shown in the blog
    
    % Main results figure (5 examples per digit)
    figure('Name', 'Linear Diffusion Results', 'Position', [100, 100, 1200, 600]);
    
    for digit = 0:9
        digit_indices = find(strcmp(labels, num2str(digit)));
        
        for example = 1:min(5, length(digit_indices))
            subplot_idx = digit * 5 + example;
            subplot(10, 5, subplot_idx);
            
            img_idx = digit_indices(example);
            imshow(squeeze(generated_images(img_idx, :, :)), []);
            
            if example == 1
                ylabel(sprintf('"%d"', digit), 'FontSize', 12, 'FontWeight', 'bold');
            end
            
            axis off;
        end
    end
    
    sgtitle('Digits generated from Linear Diffusion', 'FontSize', 16, 'FontWeight', 'bold');
end

function demonstrate_latent_space_properties(ld, sample_images)
    % Demonstrate latent space properties as described in the blog
    
    fprintf('Demonstrating latent space properties...\n');
    
    % Show original vs reconstructed (blog section: "Decoding our latents back to digits")
    figure('Name', 'PCA Reconstruction', 'Position', [150, 150, 1000, 400]);
    
    num_examples = min(10, size(sample_images, 1));
    
    % Encode using existing encoder
    [num_images, height, width] = size(sample_images);
    image_vectors = reshape(sample_images, num_images, height * width);
    
    % Apply existing standardization
    standardized_data = (image_vectors - ld.image_encoder.scaler_mean) ./ ld.image_encoder.scaler_std;
    
    % Apply existing PCA
    encoded = (standardized_data - ld.image_encoder.pca_mean) * ld.image_encoder.pca_components;
    
    % Decode back
    reconstructed = decode_from_latent(ld, encoded);
    
    for i = 1:num_examples
        % Original
        subplot(2, num_examples, i);
        imshow(squeeze(sample_images(i, :, :)), []);
        if i == 1, ylabel('Original', 'FontSize', 12); end
        title(sprintf('Image %d', i));
        axis off;
        
        % Reconstructed
        subplot(2, num_examples, i + num_examples);
        imshow(squeeze(reconstructed(i, :, :)), []);
        if i == 1, ylabel('Reconstructed', 'FontSize', 12); end
        axis off;
    end
    
    sgtitle('From 12x12 back to 28x28 (PCA Reconstruction)', 'FontSize', 14);
end

function demonstrate_noise_effects(ld, sample_images)
    % Demonstrate noise effects in latent space as shown in blog
    
    fprintf('Demonstrating noise effects in latent space...\n');
    
    num_examples = min(5, size(sample_images, 1));
    
    % Encode sample images using existing encoder
    [num_images, height, width] = size(sample_images);
    image_vectors = reshape(sample_images, num_images, height * width);
    
    % Apply existing standardization
    standardized_data = (image_vectors - ld.image_encoder.scaler_mean) ./ ld.image_encoder.scaler_std;
    
    % Apply existing PCA
    encoded = (standardized_data - ld.image_encoder.pca_mean) * ld.image_encoder.pca_components;
    
    % Different noise levels as shown in blog
    noise_levels = [0, 0.5, 1.0, 2.0];
    
    figure('Name', 'Noise Effects in Latent Space', 'Position', [200, 200, 1000, 600]);
    
    for noise_idx = 1:length(noise_levels)
        noise_std = noise_levels(noise_idx);
        
        if noise_std == 0
            noisy_encoded = encoded;
            title_prefix = 'Original';
        else
            rng(42); % Fixed seed for reproducibility
            noise = randn(size(encoded)) * noise_std;
            noisy_encoded = encoded + noise;
            title_prefix = sprintf('Noise σ=%.1f', noise_std);
        end
        
        reconstructed = decode_from_latent(ld, noisy_encoded);
        
        for img_idx = 1:num_examples
            subplot_idx = (noise_idx - 1) * num_examples + img_idx;
            subplot(length(noise_levels), num_examples, subplot_idx);
            
            imshow(squeeze(reconstructed(img_idx, :, :)), []);
            
            if img_idx == 1
                ylabel(title_prefix, 'FontSize', 10);
            end
            
            axis off;
        end
    end
    
    sgtitle('Adding noise in the latent space makes these images blurry rather than "snowy"', 'FontSize', 12);
    
    % Pure noise visualization (blog section: "This is just pure noise")
    demonstrate_pure_noise_reconstruction(ld);
end

function demonstrate_pure_noise_reconstruction(ld)
    % Show pure noise reconstruction as in the blog
    
    fprintf('Demonstrating pure noise reconstruction...\n');
    
    figure('Name', 'Pure Noise Reconstruction', 'Position', [250, 250, 800, 400]);
    
    % Generate pure noise in latent space
    rng(123);
    latent_dim = ld.LATENT_SIZE * ld.LATENT_SIZE;
    pure_noise = randn(10, latent_dim);
    
    % Decode pure noise
    noise_reconstructed = decode_from_latent(ld, pure_noise);
    
    for i = 1:10
        subplot(2, 5, i);
        imshow(squeeze(noise_reconstructed(i, :, :)), []);
        title(sprintf('Noise %d', i), 'FontSize', 10);
        axis off;
    end
    
    sgtitle('This is just pure noise, still looks surprisingly close to digits', 'FontSize', 12);
end

%% 7. Main Execution Function
function main()
    fprintf('=== Linear Diffusion MATLAB Implementation ===\n');
    fprintf('Based on Count Bayesie blog post\n\n');
    
    try
        % Load MNIST data
        fprintf('Loading MNIST data...\n');
        [train_images, train_labels, test_images, test_labels] = load_mnist_data();
        
        % Combine training and test data as mentioned in the blog
        all_imgs = cat(1, train_images, test_images);
        all_labels = [train_labels; test_labels];
        
        fprintf('Total samples: %d\n', length(all_labels));
        
        % Create and train Linear Diffusion model
        ld = LinearDiffusion();
        
        % Train the model by calling fit_model and updating ld
        ld = fit_model(ld, all_labels, all_imgs);
        
        % Generate images as shown in the blog
        fprintf('\nGenerating images for each digit...\n');
        
        % Generate 5 examples of each digit (as shown in blog results)
        test_labels_expanded = {};
        for i = 0:9
            for j = 1:5
                test_labels_expanded{end+1} = num2str(i);
            end
        end
        
        generated_images = predict_images(ld, test_labels_expanded, 137);
        
        % Visualize results
        visualize_blog_results(generated_images, test_labels_expanded);
        
        % Demonstrate key concepts from the blog
        demonstrate_latent_space_properties(ld, all_imgs(1:100, :, :));
        demonstrate_noise_effects(ld, all_imgs(1:10, :, :));
        
        fprintf('\nLinear Diffusion demo completed successfully!\n');
        
    catch ME
        fprintf('Error in Linear Diffusion demo: %s\n', ME.message);
        fprintf('Error occurred in: %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        rethrow(ME);
    end
end

%% Execute the main function
main();