%% White-balance model class
%
% Copyright (c) 2018-present, Mahmoud Afifi
% York University, Canada
% mafifi@eecs.yorku.ca | m.3afifi@gmail.com
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
% All rights reserved.
%
% Please cite the following work if this program is used:
% Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown,
% "When color constancy goes wrong: Correcting improperly white-balanced
% images", CVPR 2019.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

classdef WBmodel
    properties
        features     % training features
        mappingFuncs % training mapping functions
        K % K value for KNN
        encoder      % autoEnc or PCA object
        gamut_mapping % mapping inside the gamut (=1 for scaling, =2 for 
        % clipping). Our results reported using clipping; however, we found
        % scaling gives compelling results with over-saturated examples
    end
    methods
        function feature = encode(obj,hist) 
            % Generates a compacted feature of a given RGB-uv histogram
            % tensor. %
            feature =  obj.encoder.encode(hist);
        end
      
        function hist = RGB_UVhist(obj,I)  
            % Computes an RGB-uv histogram tensor. %
            I = im2double(I);
            if size(I,1)*size(I,2) > 202500 % if it is larger than 450*450
                factor = sqrt(202500/(size(I,1)*size(I,2))); % rescale factor
                newH = floor(size(I,1)*factor); % new height 
                newW = floor(size(I,2)*factor); % new width 
                I = imresize(I,[newH,newW],'nearest'); % resize image
            end

            h= sqrt(max(size(obj.encoder.weights,1),...
                size(obj.encoder.weights,2))/3);

            eps= 6.4/h; 
            I=(reshape(I,[],3));
            A=[-3.2:eps:3.19]; % dummy vector
            hist=zeros(size(A,2),size(A,2),3); % histogram will be stored here
            i_ind=I(:,1)~=0 & I(:,2)~=0 & I(:,3)~=0; 
            I=I(i_ind,:); % remove zereo pixels
            Iy=sqrt(I(:,1).^2+I(:,2).^2+I(:,3).^2); % intensity vector
            for i = 1 : 3 % for each color channel, do 
                r = setdiff([1,2,3],i); % exclude the current color channel
                Iu=log((I(:,i))./(I(:,r(1)))); % current color channel / the first excluded channel
                Iv=log((I(:,i))./(I(:,r(2)))); % current color channel / the second excluded channel
                diff_u=abs(Iu-A); % differences in u space
                diff_v=abs(Iv-A); % differences in v space
                % for old Matlab versions:
                % diff_u=abs(repmat(Iu,[1,size(A,2)])-repmat(A,[size(Iu,1),1]));
                % diff_v=abs(repmat(Iv,[1,size(A,2)])-repmat(A,[size(Iv,1),1]));
                % here, we will use a matrix multiplication expression to compute eq. 4 in the main paper.
                diff_u=(reshape((reshape(diff_u,[],1)<=eps/2),...
                    [],size(A,2))); % don't count any pixel has difference beyond the threshold in the u space
                diff_v=(reshape((reshape(diff_v,[],1)<=eps/2),...
                    [],size(A,2))); % similar in the v space
                hist(:,:,i)=(Iy.*double(diff_u))'*double(diff_v); % compute the histogram 
                % for old Matlab versions: 
                % hist(:,:,i)=(repmat(Iy, [1, size(diff_u,2)]).* double(diff_u))'*double(diff_v); % compute the histogram 
                hist(:,:,i)=sqrt(hist(:,:,i)/sum(sum(hist(:,:,i)))); % sqrt the histogram after normalizing
            end
            hist = imresize(hist,[h h],'bilinear');
        end

        function hist = fast_RGB_UVhist(obj, I)
            % Downscample if needed
            I = im2double(I);
            if numel(I) > 202500 * 3
                scaleFactor = sqrt(202500 / (size(I,1) * size(I,2)));
                I = imresize(I, scaleFactor, 'nearest');
            end
            
            % Flatten and remove zero
            I = reshape(I, [], 3);
            nonzero_idx = all(I > 0, 2);
            I = I(nonzero_idx, :);

            % Randomly subsample fraction of remaining pixels
            subsample_fraction = 0.10; 

            N = size(I, 1);
            if N > 0
                keep_mask = (rand(N, 1) < subsample_fraction);
                I = I(keep_mask, :);
            end
            
            % Prepare bin centers
            h = sqrt(max(size(obj.encoder.weights,1), ...
                         size(obj.encoder.weights,2)) / 3);
            epsVal = 6.4 / h;
            A = -3.2 : epsVal : 3.19;   
            K = numel(A);                % # bin centers
            hist = zeros(K, K, 3);       % final histogram (KxK) for each channel
            
            Iy = sqrt(sum(I.^2, 2));  % Nx1
            
            for iChan = 1 : 3
                
                r = setdiff([1,2,3], iChan);
                Iu = log(I(:, iChan) ./ I(:, r(1)));
                Iv = log(I(:, iChan) ./ I(:, r(2)));
                
                diff_u = abs(Iu - A);   % NxK
                diff_v = abs(Iv - A);   % NxK
                
                boolean_u = (diff_u <= (epsVal / 2));
                boolean_v = (diff_v <= (epsVal / 2));
                            
                bu_weighted = (Iy .* double(boolean_u)); 
                hist_chan = (bu_weighted') * double(boolean_v);  % KxK
                
                sum_bins = sum(hist_chan(:));
                if sum_bins > 0
                    hist_chan = sqrt(hist_chan / sum_bins);
                end
                
                % Place in the 3D histogram
                hist(:,:,iChan) = hist_chan;
            end

            for c = 1 : 3
                hist(:,:,c) = imresize(hist(:,:,c), [h, h], 'bilinear');
            end
        end

        function [corrected, mf, in_gamut, dynamic_k, fallback_used] = correctImage(obj, I, feature, sigma, use_dynamic_k, use_fallback, use_fasthist)
            I = im2double(I);
            fallback_used = false; % default
            
            % Defaults for omitted arguments
            if nargin < 7, use_fasthist  = false;  end
            if nargin < 6, use_fallback  = false;  end
            if nargin < 5, use_dynamic_k = false;  end
            if nargin < 4 || isempty(sigma)
                sigma = 0.25;
            end

            % If no feature provided, compute it
            if nargin < 3 || isempty(feature)
                if use_fasthist
                    % Use subsampled fasthist
                    hist = obj.fast_RGB_UVhist(I);
                else
                    % Use original histogram approach
                    hist = obj.RGB_UVhist(I);
                end
                feature = obj.encode(hist);
            end

         
            % -----------------------------------------------------
            % IF FALLBACK ENABLED
            % -----------------------------------------------------
            if use_fallback
                meanRGB = mean(reshape(I, [], 3), 1);      
                % Ratio between largest and smallest channel
                ratio = max(meanRGB) / max(min(meanRGB), 1e-8); 

                if ratio > 30
                    % Shades of Gray fallback
                    p_val = 8;
                    %fprintf('Fallback triggered. Ratio=%.2f\n', ratio);
                    fallback_used = true;
    
                    fallback_im = obj.doShadeGrayFallback(I, p_val);
                    %fallback_im = obj.neutralize_lab_ab(fallback_im, 0.1);
                    %fallback_im = obj.doGrayEdgeFallback(I, p_val);
    
                    base_alpha = 0.75;
                    alpha = max(0.5, base_alpha * (20 / ratio));
    
                    fallback_im = obj.deSat(fallback_im, alpha);
    
                    % Return early with fallback image
                    corrected = fallback_im; 
                    mf = [];
                    in_gamut = [];
                    dynamic_k = NaN;
                    return;
                end      
            end

            % -----------------------------------------------------
            % Do original KNN Approach
            % -----------------------------------------------------
            
            % KNN search
            default_k = obj.K;
            max_dynamic_k = round(obj.K * 4.0); % Max allowed neighbors
            
            [dH, idH] = pdist2(obj.features, feature, 'euclidean', 'Smallest', max_dynamic_k);
            
            % Dynamically adjust neighbor selection
            if use_dynamic_k
                max_k = default_k; % Maximum number of neighbors
                base_threshold = 0.001;
            
                % Compute dynamic K by looking at variations
                [dynamic_k, cv] = obj.selectDynamicK(dH, idH, max_k, base_threshold, I);
                
                % Use only the top dynamic_k neighbors
                dH = dH(1:dynamic_k);
                idH = idH(1:dynamic_k);
            else
                dynamic_k = default_k; % default val (25)
            end

            % Compute weights using RBF
            weightsH = exp(-((dH).^2) / (2 * sigma^2)); % Compute weights
            weightsH = weightsH / sum(weightsH); % Normalize weights

            % Blend nearest mapping functions
            M = size(obj.mappingFuncs, 2);
            mf = sum(weightsH .* obj.mappingFuncs(idH, :), 1); % Blend mapping functions
            mf = reshape(mf, [M / 3, 3]); % Reshape to 11x3
            
            % Apply the correction matrix
            [corrected, in_gamut] = obj.color_correction(I, mf, obj.gamut_mapping);
            corrected = double(corrected);
        end

        
        function [out,map] = color_correction(obj,input, m, gamut_map)
            % Applies a mapping function m to a given input image. %
            if nargin == 3
                gamut_map = 2;
                map = [];
            end
            sz=size(input);
            input=reshape(input,[],3);
            if gamut_map == 1
                input_ = input; % take a copy--will be used later
            end
            input=obj.kernelP(input); % raise it to a higher degree (Nx11)
            out=input * m; 
            if gamut_map == 1 % if scaling,
                out = obj.norm_scaling(input_, out);
                map = [];
            elseif gamut_map == 2 % if clipping,
                [out,map] = obj.out_of_gamut_clipping(out);
            end
            out=reshape(out,[sz(1),sz(2),sz(3)]); % reshape it from Nx3 to the original size
        end
        
        
        function [I,map] = out_of_gamut_clipping(obj,I)
            % Clips out-of-gamut pixels. %
            I = im2double(I);
            map = ones(size(I)); % in-gamut map
            map(I>1) = 0;
            map(I<0) = 0;
            map = map(:,1) & map(:,2) & map(:,3);
            I(I>1) = 1; % any pixel is higher than 1, clip it to 1
            I(I<0) = 0; % any pixel is below 0, clip it to 0
        end
        
        function [I_corr] = norm_scaling(obj, I, I_corr)
            % Scales each pixel based on original image energy. %
            norm_I_corr = sqrt(sum(I_corr.^2,2));
            inds = norm_I_corr ~= 0;
            norm_I_corr = norm_I_corr(inds);
            norm_I = sqrt(sum(I(inds,:).^2,2));
            I_corr(inds, :) = I_corr(inds,:)./norm_I_corr .* norm_I;
        end

        function O=kernelP(obj,I)
            % kernel(R,G,B)=[R,G,B,RG,RB,GB,R2,G2,B2,RGB,1];
            % Kernel func reference:
            % Hong, et al., "A study of digital camera colorimetric 
            % characterization based on polynomial modeling." Color 
            % Research & Application, 2001.
            O=[I,... %r,g,b
                I(:,1).*I(:,2),I(:,1).*I(:,3),I(:,2).*I(:,3),... %rg,rb,gb
                I.*I,... %r2,g2,b2
                I(:,1).*I(:,2).*I(:,3),... %rgb
                ones(size(I,1),1)]; %1
        end  

        %% Added Functions

        function [dynamic_k, cv] = selectDynamicK(obj, distances, neighborIDs, default_k, ~, I)
            % old experimental function to vary # of neighbors used 
            % depending on coefficient of variation 
        
            % get CV
            mean_d = mean(distances);
            std_d  = std(distances);
            cv = std_d / (mean_d + eps);

            min_k = ceil(default_k / 2);   
            max_k = round(default_k * 1.0);    
        

            lower_threshold = 0.08;   
            upper_threshold = 0.20;  
        
            % linear interpolate
            if cv < lower_threshold
                ratio = (lower_threshold - cv) / lower_threshold; 
                dynamic_k = round(default_k + ratio * (max_k - default_k));
        
            elseif cv > upper_threshold
                max_cv = 1.0;  
                ratio = (min(cv, max_cv) - upper_threshold) / (max_cv - upper_threshold);
                dynamic_k = round(default_k - ratio * (default_k - min_k));
            else
                dynamic_k = default_k;
            end
        
            % Clamp final val
            dynamic_k = max(min_k, min(dynamic_k, max_k));
        
            % debug
            % fprintf('CV=%.3f => dynamic_k=%d (range=[%d,%d])\n', cv, dynamic_k, min_k, max_k);
        end

        

        function fallback_im = doGrayEdgeFallback(obj, I, p_val)
            % Applies gray-edge correction to image I with exponent p_val

            I = im2double(I);
            [H, W, ~] = size(I);
        
            % Separate color channels
            R = I(:,:,1); 
            G = I(:,:,2); 
            B = I(:,:,3);
        
            % Compute derivatives
            fx = [-1, 1];  % horizontal derivative kernel
            fy = [-1; 1];  % vertical derivative kernel
        
            R_x = conv2(R, fx, 'same');  R_y = conv2(R, fy, 'same');
            G_x = conv2(G, fx, 'same');  G_y = conv2(G, fy, 'same');
            B_x = conv2(B, fx, 'same');  B_y = conv2(B, fy, 'same');
        
            % Gradient magnitude
            R_mag = sqrt(R_x.^2 + R_y.^2);
            G_mag = sqrt(G_x.^2 + G_y.^2);
            B_mag = sqrt(B_x.^2 + B_y.^2);
        
            % p-th power sum
            R_sum_p = sum(R_mag(:).^p_val);
            G_sum_p = sum(G_mag(:).^p_val);
            B_sum_p = sum(B_mag(:).^p_val);
        
            numPix = H*W;  
        
            % p-th means
            R_mean_p = (R_sum_p / numPix)^(1/p_val);
            G_mean_p = (G_sum_p / numPix)^(1/p_val);
            B_mean_p = (B_sum_p / numPix)^(1/p_val);
        
            % Scale
            overall_mean_p = (R_mean_p + G_mean_p + B_mean_p) / 3;
        
            scale_R = overall_mean_p / max(R_mean_p, 1e-8);
            scale_G = overall_mean_p / max(G_mean_p, 1e-8);
            scale_B = overall_mean_p / max(B_mean_p, 1e-8);
        
            % Apply channel scaling
            fallback_im = I;
            fallback_im(:,:,1) = fallback_im(:,:,1) * scale_R;
            fallback_im(:,:,2) = fallback_im(:,:,2) * scale_G;
            fallback_im(:,:,3) = fallback_im(:,:,3) * scale_B;
        
            % Clip
            fallback_im = max(min(fallback_im, 1), 0);
        end

        function fallback_im = doShadeGrayFallback(obj, I, p_val)
            % Do shades of grey correction on image
            I = im2double(I);
            R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
            
            % Compute sum of each channel ^p
            R_sum_p = sum(R(:).^p_val);
            G_sum_p = sum(G(:).^p_val);
            B_sum_p = sum(B(:).^p_val);
            
            % Get total # pixels
            numPix = numel(R);
            
            % Compute p-th root of average of channel^p
            R_mean_p = (R_sum_p / numPix)^(1/p_val);
            G_mean_p = (G_sum_p / numPix)^(1/p_val);
            B_mean_p = (B_sum_p / numPix)^(1/p_val);
            
            overall_mean_p = (R_mean_p + G_mean_p + B_mean_p)/3;
            
            scale_R = overall_mean_p / max(R_mean_p, 1e-8);
            scale_G = overall_mean_p / max(G_mean_p, 1e-8);
            scale_B = overall_mean_p / max(B_mean_p, 1e-8);
            
            % Scale each channel so p-norms are equal
            fallback_im = I;
            fallback_im(:,:,1) = fallback_im(:,:,1) * scale_R;
            fallback_im(:,:,2) = fallback_im(:,:,2) * scale_G;
            fallback_im(:,:,3) = fallback_im(:,:,3) * scale_B;
            
            % Clip to [0,1]
            fallback_im = max(min(fallback_im, 1), 0);
                 
        end

        function fallback_im = doWhitePatchFallback(obj, I)
            % Applies white patch white balancing to image

            I = im2double(I);
        
            % Compute the max pixel val for each channel
            max_R = max(reshape(I(:,:,1), [], 1));
            max_G = max(reshape(I(:,:,2), [], 1));
            max_B = max(reshape(I(:,:,3), [], 1));
        
            % Get overall average 
            overall_max = (max_R + max_G + max_B) / 3;
        
            % Compute scaling factors for each channel.
            eps_val = 1e-8;
            scale_R = overall_max / max(max_R, eps_val);
            scale_G = overall_max / max(max_G, eps_val);
            scale_B = overall_max / max(max_B, eps_val);
        
            % Apply scaling to each channel
            fallback_im = I;
            fallback_im(:,:,1) = I(:,:,1) * scale_R;
            fallback_im(:,:,2) = I(:,:,2) * scale_G;
            fallback_im(:,:,3) = I(:,:,3) * scale_B;
        
            % Clamp
            fallback_im = max(min(fallback_im, 1), 0);
        end

        function desat_im = deSat(obj, I, alpha)
            % Desaturates an image to 1 * alpha of the original saturation

            hsv_im = rgb2hsv(I);

            % Desaturate and clamp
            hsv_im(:,:,2) = hsv_im(:,:,2) * alpha;
            hsv_im(:,:,2) = max(min(hsv_im(:,:,2), 1), 0);
            desat_im = hsv2rgb(hsv_im);
        end

        function output = neutralize_lab_ab(obj, im, alpha)
        % Reduces green/yellow cast in LAB space by shifting a*, b* toward 0
        
            if isa(im, 'uint8')
                im = im2double(im);
            end
        
            % Convert to LAB color space
            lab = rgb2lab(im);
        
            % Pull green and yellow channels down
            lab(:,:,2) = lab(:,:,2) * (1 - alpha);  % a*
            lab(:,:,3) = lab(:,:,3) * (1 - alpha);  % b*
        
            % Convert back to RGB
            output = lab2rgb(lab);
        
            % Clamp 
            output = max(min(output, 1), 0);
        end
    end
end
