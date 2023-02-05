
%====================================================================================================================================================

%{

cd( '/net/s8iddata/export/8-id-ECA/Analysis/atripath/ASTRA_3D_lamino_simulation/' )

%}

%========

clear; close all

restoredefaultpath; 

%========

addpath( genpath( '/net/s8iddata/export/8-id-ECA/Analysis/atripath/astra/matlab/' ));                     % where ASTRA is installed
addpath( genpath( '/net/s8iddata/export/8-id-ECA/Analysis/atripath/ASTRA_3D_lamino_simulation/code/' ));  % where the code demo is located
    
%========

sol.gpu_id = 1; 

reset( gpuDevice( sol.gpu_id )); 

%========================
% Define 3D sample volume
%========================

cube = setup_3dsamplevolume( );

[ Nr, Nc, Np ] = size( cube ); % get final tomo problem size

%=====================
% define a 3D support?
%=====================

% cube_support = ( cube ~= 0 );
% 
% [ cube_support ] = lpf_gauss( cube_support, 0.20 * size( cube_support ) );
% cube_support     = ( abs( cube_support ) > 1e-3 );

%=============================
% projection angles in radians
%=============================

Nrot = 100;

%========

% angles = linspace2( -pi * 45 / 180, +pi * 45 / 180, Nrot );
% angles = linspace2( 0, +1 * pi, Nrot );

angles = linspace2( 0, +2 * pi, Nrot );

% angles = linspace( -1 * pi, +1* pi, Nrot );

% angles = 0;

%========

% % theta_wedge = pi/2.0;   % 90 deg
% % theta_wedge = pi/2.4;   % 75 deg
% % theta_wedge = pi/3.0;   % 60 deg
% 
% % theta_wedge = pi/4.0;   % 45 deg
% % theta_wedge = pi/6.0;   % 30 deg
% 
% % theta_wedge = pi/9.0;   % 20 deg 
% % theta_wedge = pi/12.0;  % 15 deg 
% % theta_wedge = pi/18.0;  % 10 deg 

% angles = linspace( -theta_wedge, +theta_wedge, Nrot);
% % angles = [ angles, linspace( pi - theta_wedge, pi + theta_wedge, Nrot + 10 ) ];
% 
% % angles = [ linspace( -theta_wedge, +theta_wedge, Nrot ), linspace( pi - theta_wedge, pi + theta_wedge, Nrot ) ];

%========

angles = single( angles );

%=================
% axis of rotation
%=================

aor = [ -0.0; -0.105; +1 ]; % ( y, x, z )    % ~6 degrees
% aor = [ -0.0; -0.0175; +1 ]; % ( y, x, z )    % ~1 degrees

aor = single( aor / norm( aor ));    % rescale to unit vector

%====================================================================================================================================================
%                                                       ASTRA Sinogram Creation
%====================================================================================================================================================

% define the direction of source, detector vs rotation angles vs axis of rotation
vectors   = setup_parallel3dvec( angles, aor );

proj_geom = astra_create_proj_geom( 'parallel3d_vec', Np, Nr, vectors );

vol_geom = astra_create_vol_geom( [ Nc, Nr, Np ] );    % assumes ( x, y, z )

[ proj_id, proj_data ]  = astra_create_sino3d_cuda( cube, proj_geom, vol_geom );    % Create projection data (i.e. sinogram) from this

%====================================================================================================================================================
%                                                       By-hand Sinogram Creation
%====================================================================================================================================================

cor       = single( 0.5 * [ Nr, Nc, Np ] + 0.5 );     % ( y, x, z )
rotvol_sz = single( [ Nr, Nc, Np ] );                 % ( y, x, z )

% rotvol_interp = 'nearest';
rotvol_interp = 'linear';
% rotvol_interp = 'cubic';
% rotvol_interp = 'makima';
% rotvol_interp = 'spline';

rotvol_fill  = single( 0.0 );

%========

proj_data_rot = zeros( [ Nr, Nrot, Np ], 'single' );

%========

cor           = gpuArray( cor );
aor           = gpuArray( aor );
rotvol_sz     = gpuArray( rotvol_sz );
proj_data_rot = gpuArray( proj_data_rot );
angles        = gpuArray( angles );
rotvol_fill   = gpuArray( rotvol_fill );
cube          = gpuArray( cube );

%========================================================
% Construct sinogram for manual 3D rotation without ASTRA
%========================================================

for ii = 1 : Nrot
    
    [ V, ~, ~, ~ ] = rotate_volume_interp3( cube, cor, aor, angles( ii ), rotvol_sz, rotvol_interp, rotvol_fill );
    
    proj_data_rot( :, ii, : ) = squeeze( sum( V, 2 ));

end

proj_data_rot = gather( proj_data_rot );

%==============================================================
% uncomment this to get the plots I included (this part is slow)
%==============================================================

%{

for ii = 1 : 5 : Nrot

    tmp0 = squeeze( proj_data( :, ii, : ) );
    tmp1 = squeeze( proj_data_rot( :, ii, : ) );

    %========

    h1 = figure( 666 );
    set( h1, 'Visible', 'off', 'Position',[ 1, 1, 1920, 1080 ] )
    
    subplot(231); imagesc( transpose( tmp0 )); 
    xlabel('+y'); ylabel('+z'); colormap( 'bone'); 
    set( gca, 'ydir', 'normal' ); colorbar; title('ASTRA projection'); %daspect([1 1 1]); 
    
    subplot(232); imagesc( transpose( tmp1 )); 
    xlabel('+y'); ylabel('+z'); colormap( 'bone'); 
    set( gca, 'ydir', 'normal' ); colorbar; title('Manual 3D rotation projection'); %daspect([1 1 1]); 
    
    subplot(233); imagesc( transpose( tmp0 - tmp1 )); 
    xlabel('+y'); ylabel('+z'); colormap( 'bone'); set( gca, 'ydir', 'normal' ); colorbar; title('ASTRA - manual'); %daspect([1 1 1])

    %========

    [ V, ~, ~, ~ ] = rotate_volume_interp3( cube, cor, aor, angles( ii ), rotvol_sz, rotvol_interp, rotvol_fill );

    V3dplt = imresize3( gather( V ), 1.00 );
    
    subplot( 2, 3, 4 );
    params.isoslvl = 1e-3;
    params.alphalvl = 0.35;
    isosurface_phase_surface_raw( V3dplt, params ); % colormap gray
    % daspect([1 1 1])
    view( 0, 0 )
    title('view along +y')
    
    subplot( 2, 3, 5 );
    params.isoslvl = 1e-3;
    params.alphalvl = 0.35;
    isosurface_phase_surface_raw( V3dplt, params ); % colormap gray
    % daspect([1 1 1])
    view(+90,0)
    title('view along -x')
    
    subplot( 2, 3, 6 );
    params.isoslvl = 1e-3;
    params.alphalvl = 0.35;
    isosurface_phase_surface_raw( V3dplt, params ); % colormap gray
    view(+60, +70)
    % daspect([1 1 1])
    
    drawnow
    
    export_fig( num2str( angles( ii ), 'abs_phs_2Dprojection_%0.3f.png' ), '-r120.0' )
    close all;
        
end

%}

%====================================================================================================================================================
%                                                   PERFORM TOMOGRAPHIC RECONSTRUCTION
%====================================================================================================================================================

% use the manual 3D rotation sinogram?

proj_id = astra_mex_data3d( 'create', '-proj3d', proj_geom, proj_data_rot );  
astra_mex_data3d( 'store', proj_id, proj_data_rot );

%========

% Create a data object for the reconstruction
rec_id = astra_mex_data3d( 'create', '-vol', vol_geom );

% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct( 'SIRT3D_CUDA' );
% cfg = astra_struct( 'CGLS3D_CUDA' );
% cfg = astra_struct( 'BP3D_CUDA' );

cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId     = proj_id;

%==================================
% Create a data object for the mask
%==================================

% mask_id = astra_mex_data3d( 'create', '-vol', vol_geom, cube_support );
% cfg.option.ReconstructionMaskId = mask_id;

% for SIRT3D only
cfg.option.MinConstraint = 0;
cfg.option.MaxConstraint = 1.0;

%========

% Create the algorithm object from the configuration structure
alg_id = astra_mex_algorithm( 'create', cfg );

for rr = 1 : 1
    
    %========
    
    astra_mex_algorithm( 'iterate', alg_id, 25 );
%     astra_mex_algorithm( 'run', alg_id );

%     %========
%     
%     cost( rr ) = astra_mex_algorithm('get_res_norm', alg_id ) ;
%     
%     %========
%     
%     % Get the result
%     rec = astra_mex_data3d( 'get', rec_id );
%     
%     rec( rec > 1 ) = 1;
%     rec( rec < 0 ) = 0;
%     
% %     rec = rec .* cube_support;
%     
%     astra_mex_data3d( 'set', rec_id, rec );
%     
%     %========
    
end

%========

% Get the result
rec = astra_mex_data3d( 'get', rec_id );

%========

figure;
sp4 = subplot(231); imagesc( squeeze( sum( cube, 1 )) / size( cube, 1 ) ); 
title('y projection, gt'); 
colormap( sp4, 'bone')
colorbar
daspect([1 1 1])

sp5 = subplot(232); imagesc( squeeze( sum( cube, 2 )) / size( cube, 2 ) ); 
title('x projection, gt'); 
colormap( sp5, 'bone')
colorbar
daspect([1 1 1])

sp6 = subplot(233); imagesc( squeeze( sum( cube, 3 )) / size( cube, 3 ) ); 
title('z projection, gt'); 
colormap( sp6, 'bone')
colorbar
daspect([1 1 1])

sp1 = subplot(234); imagesc( squeeze( sum( rec, 1 )) / size( rec, 1 ) ); 
title('y projection, rec'); 
colormap( sp1, 'bone')
colorbar
daspect([1 1 1])

sp2 = subplot(235); imagesc( squeeze( sum( rec, 2 )) / size( rec, 2 ) ); 
title('x projection, rec'); 
colormap( sp2, 'bone')
colorbar
daspect([1 1 1])

sp3 = subplot(236); imagesc( squeeze( sum( rec, 3 )) / size( rec, 3 ) ); 
title('z projection, rec'); 
colormap( sp3, 'bone')
colorbar
daspect([1 1 1])

%========

figure;
sp4 = subplot(231); imagesc( squeeze( cube( round(end/2), :, : ) ))
title('y slice, gt'); 
colormap( sp4, 'bone')
colorbar
daspect([1 1 1])

sp5 = subplot(232); imagesc( squeeze( cube( :, round(end/2), : ) ))
title('x slice, gt'); 
colormap( sp5, 'bone')
colorbar
daspect([1 1 1])

sp6 = subplot(233); imagesc( squeeze( cube( :, :, round(end/2) ) )) 
title('z slice, gt'); 
colormap( sp6, 'bone')
colorbar
daspect([1 1 1])

sp1 = subplot(234); imagesc( squeeze( rec( round(end/2), :, : ) ))
title('y slice, rec'); 
colormap( sp1, 'bone')
colorbar
daspect([1 1 1])

sp2 = subplot(235); imagesc( squeeze( rec( :, round(end/2), : ) ))
title('x slice, rec'); 
colormap( sp2, 'bone')
colorbar
daspect([1 1 1])

sp3 = subplot(236); imagesc( squeeze( rec( :, :, round(end/2) ) ))
title('z slice, rec'); 
colormap( sp3, 'bone')
colorbar
daspect([1 1 1])

%========

% Clean up. Note that GPU memory is tied up in the algorithm object, and main RAM in the data objects.
astra_mex_algorithm( 'delete', alg_id  );
astra_mex_data3d(    'delete', rec_id  );
astra_mex_data3d(    'delete', proj_id );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vectors = setup_parallel3dvec( angles, aor ) 

%     p_source = [ +0; -1; +0 ];  % ( y, x, z ), location of center of parallel beam
%     p_cod    = [ +0; +1; +0 ];   % ( y, x, z ), location of center of detector
    p_source = [ +0; +1; +0 ];  % ( y, x, z ), location of center of parallel beam
    p_cod    = [ +0; -1; +0 ];   % ( y, x, z ), location of center of detector

    p_det_horiz = [ +1; +0; +0 ];     % ( y, x, z ), direction of detector horizontal 
    p_det_vert  = [ +0; +0; +1 ];     % ( y, x, z ), direction of detector vertical


    vectors = zeros( numel( angles ), 12 );

    for aa = 1 : numel( angles )

        %================
        % rotation matrix
        %================

        Rrot = create_3Drotmatrix( angles( aa ), [ 0, 0, 0 ], aor );

        Rrot = Rrot( 1 : 3, 1 : 3 );

        %==============
        % ray direction
        %==============

        prot = Rrot * p_source;

        vectors( aa, 1 ) = prot( 1 );     
        vectors( aa, 2 ) = prot( 2 );    
        vectors( aa, 3 ) = prot( 3 );	 

        %==================
        % center of detector
        %==================

        prot = Rrot * p_cod;

        vectors( aa, 4 ) = prot( 1 );
        vectors( aa, 5 ) = prot( 2 );
        vectors( aa, 6 ) = prot( 3 );

        %========================================================
        % vectors to two points in detector defining its 2D plane
        %========================================================

        p_det_horiz_rot = Rrot * p_det_horiz; 
        p_det_vert_rot  = Rrot * p_det_vert;

        % vector from detector pixel ( 0, 0, 0 ) to ( 1, 0, 0 )
        vectors( aa, 7 ) = p_det_horiz_rot( 1 );
        vectors( aa, 8 ) = p_det_horiz_rot( 2 );
        vectors( aa, 9 ) = p_det_horiz_rot( 3 );

        % vector from detector pixel ( x, y, z ) ?= ( 0, 0, 0) to ( 0, 0, 1 )
        vectors( aa, 10 ) = p_det_vert_rot( 1 ); 
        vectors( aa, 11 ) = p_det_vert_rot( 2 );        
        vectors( aa, 12 ) = p_det_vert_rot( 3 );          % z 

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function V = setup_3dsamplevolume

    %==========================
    % 1 : 1 Aspect Ratio Sample
    %==========================
    
    Nr = 4 * 96;
    Nc = 4 * 128;
    Np = 4 * 48;
    
    V = zeros( Nr, Nc, Np, 'single' );
    
    V( round( (( 0.5 * Nr + 1 ) - 0.30 * Nr ) : ( 0.5 * Nr + 0.30 * Nr )),  ...
       round( (( 0.5 * Nc + 1 ) - 0.25 * Nc ) : ( 0.5 * Nc + 0.25 * Nc )), ...
       round( (( 0.5 * Np + 1 ) - 0.07 * Np ) : ( 0.5 * Np + 0.07 * Np )) ) = 1;
    
    V( 150 : 155, :, : ) = 0;
    V( 180 : 185, :, : ) = 0;
    
    V( :, 200 : 210, : ) = 0;
    V( :, 300 : 350, : ) = 0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           rotate the sample above so that its surface normal is parallel to lamino axis of rotation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    rot_angle = -atan( 0.105 );     % ~6 degrees
    % rot_angle = -atan( 0.0175 );    % ~1 degrees
    
    %===================
    % center of rotation
    %===================
    
    cor = [ 0.5 * Nr + 0.5, ...
            0.5 * Nc + 0.5, ...
            0.5 * Np + 0.5 ];     % ( y, x, z )
    
    %=============================
    % axis of rotation unit vector
    %=============================
    
    aor = [ +1.0, +0.0, +0.0 ];   % ( y, x, z )
    aor = aor / norm( aor );
    
    %========
    
    % rotvol_interp = 'nearest';
    rotvol_interp = 'linear';
    % rotvol_interp = 'cubic';
    % rotvol_interp = 'makima';
    % rotvol_interp = 'spline';
    
    rotvol_fill = single( 0.0 );
    
    %========
    
%     figure
%     params.isoslvl = 1e-6;
%     params.alphalvl = 0.35;
%     % isosurface_phase_surface( V, params ); % colormap gray
%     isosurface_phase_surface_raw( V, params );
%     daspect([1 1 1])
%     % view( 90, 0 )
    
    [ V, ~, ~, ~ ] = rotate_volume_interp3( V, cor, aor, rot_angle, [ Nr, Nc, Np ], rotvol_interp, rotvol_fill );
             
    V( abs( V ) < 1e-2 ) = 0;
                   
    % figure
    % params.isoslvl = 1e-6;
    % params.alphalvl = 0.35;
    % % isosurface_phase_surface( V, params ); % colormap gray
    % isosurface_phase_surface_raw( V, params );
    % daspect([1 1 1])
    % % view( 90, 0 )

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

