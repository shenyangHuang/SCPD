function total_dos = generate_ADOS_real(dataset_name, Nbins, seed)
    
    if nargin<2
        Nbins = 50;
    end

    rng(seed);
    fprintf("running trial %d \n", seed);



    load(strcat("processed/",dataset_name,"/",dataset_name,"_all_graphs.mat"));
    dataset.all_graphs = all_graphs;

    clear all_graphs;
    
    is_attr = isfile(strcat('processed/',dataset_name,'/',dataset_name,'_all_attributes.mat'));
    is_lbl = isfile(strcat('processed/',dataset_name,'/',dataset_name,'_all_OH_labels.mat'));
    
    total_features = 0;

    if is_lbl
        load(strcat("processed/",dataset_name,"/",dataset_name,"_all_OH_labels.mat"));
        dataset.all_OH_labels = all_OH_labels;
        clear all_labels;
        total_features = total_features + size(dataset.all_OH_labels{1},2);
    end

    
    if is_attr
        load(strcat("processed/",dataset_name,"/",dataset_name,"_all_attributes.mat"),all_attributes);
        dataset.all_attributes = all_attributes;
        clear all_attributes;
        total_features = total_features + size(dataset.all_attributes{1},2);
    end


    
    if total_features == 0
        total_features = 1;
    end
    output_features = zeros(size(dataset.all_graphs,1),(1+0.5*total_features*(total_features+1))*Nbins);


    runtimes = zeros(size(dataset.all_graphs,1), 4);
    dos_times = zeros(size(dataset.all_graphs,1));
    full_start = tic();
    for i = 1:size(dataset.all_graphs,1)
        if rem(i, round(size(dataset.all_graphs,1)/10))==0
            fprintf('.');
        end
        adjacency_matrix = dataset.all_graphs{i};
        num_nodes = size(adjacency_matrix,1);
        num_edges = nnz(adjacency_matrix);


        
        d = sparse(sum(adjacency_matrix,2));
        L = diag(d) - adjacency_matrix;
        Dhalf = diag(sparse(1./sqrt(d)));
        nL = Dhalf*(L*Dhalf);
        %nL(1:10,1:10)
        N = sparse(real(rescale_matrix(nL, 0, 2)));

        if ~is_lbl && ~is_attr % use degrees as feature if no other feature
            degrees = full(sum(adjacency_matrix))';
            feature_vectors = degrees - mean(degrees);
            total_features=1;
        else
            feature_vectors = [];
            if is_lbl
                feature_vectors = [feature_vectors double(dataset.all_OH_labels{i})];
            end
            if is_attr
                feature_vectors = [feature_vectors double(dataset.all_attributes{i} - mean(dataset.all_attributes{i},1))];
            end
        end
        local_start = tic();


        output_features(i,1:Nbins) = compute_dos(N, Nbins);

        
        j=1;
        for k = 1:total_features
            output_features(i,(Nbins*j)+1:Nbins*(j+1)) = compute_ldos(N, feature_vectors(:,k), Nbins);
            j = j+1;
        end
        
        elapsed = toc(local_start);
        dos_times(i) = elapsed;
        runtimes(i,:) = [num_nodes, num_edges, total_features, elapsed];
    end
    total_elapsed = toc(full_start);
    total_dos = sum(dos_times(:));
    fprintf("total dos time %f seconds\n", total_dos);
    
    if not(isfolder('embeddings/'+dataset_name))
        mkdir('embeddings/'+dataset_name)
    end
    
    dlmwrite("embeddings/"+dataset_name+"/"+dataset_name+"_dos"+string(seed)+".csv", real(output_features(:,1:Nbins)));
    dlmwrite("embeddings/"+dataset_name+"/"+dataset_name+"_dos_ldos"+string(seed)+".csv", real(output_features(:,1:Nbins*(1+total_features))));
end