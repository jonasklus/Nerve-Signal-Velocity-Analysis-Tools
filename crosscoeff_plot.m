function [Cross_correlations,corr_threshold] = crosscoeff_plot(Trace_distal, Trace_proximal, fs, window_width, step_size, t_singleplots);

%Alternative implementation of Correlation_coefficient.m
%Calculates cross-correlation between two channels, but does so with a
%rolling window of width and step defined in input. Both are in seconds.
%It then plots this as a heatmap (x axis: time, y axis: lead/lag, z axis:
%coefficient).
%Single_plots leads to plotting of individual corr_coeff plots around
%indicated timepoints, given as an array in seconds. Note for this to
%happen, sing_plot below must be = 1. Done with same window_width as
%heatmap plot.
%Allows for interpolation too (using cubic spline).

%Define inputs
Trace_d = Trace_distal;
Trace_p = Trace_proximal;
plot_type = 0; %Choose 1 for log on x axis (lag), or any other for linear.
interp = 0; %Choose 1 for heatmaps with interpolation, any other for none.
sing_plot = 1; %Choose 1 for single plots at provided timepoints to be created.
Interpolation_factor = 1; %Choose a value for upsampling (recommended powers of 2). 1 skips interpolation.
Plot_interpolation_check = 0; %Choose 1 to plot the pre- and post-interpolation traces to check them.

%%
%First off carry out data interpolation to input variables if necessary.
%Done first before beginning any work on cross-correlation

t = 0:(1/fs):((length(Trace_p)-1)/fs); %Define time vector

if Interpolation_factor > 1
    fs_new = fs*Interpolation_factor; %New sampling frequency scaled by interpolation factor
    t_new = [0:(1/fs_new):max(t)]; %New time vector
    Trace_d_new = spline(t,Trace_d,t_new); %New traces via spline interpolation
    Trace_p_new = spline(t,Trace_p,t_new);

    if Plot_interpolation_check == 1; %Checks whether you have requested to plot the interpolated traces
        figure
        subplot(2,1,1);
        plot(t,Trace_d,'o');hold on;plot(t_new,Trace_d_new,'.');
        subplot(2,1,2);
        plot(t,Trace_p,'o');hold on;plot(t_new,Trace_p_new,'.');
        linkaxes
    end

    fs = fs_new;    %All variables replaced by their interpolated counterparts
    t = t_new;
    Trace_d = Trace_d_new;
    Trace_p = Trace_p_new;

end

%%
%Define working variables for cross correlation portion

step_size_fs = step_size*fs; %Express step size as data points
window_width_fs = window_width*fs; %Express window width as data points
range_lags = 0.002; %Define ranges of lead/lags to be calculated and saved (in s)

Cross_correlations = NaN((1 + range_lags*fs*2) , round(length(t)/step_size_fs)); %Define output array

ajustmnt = round((length(t)/(step_size_fs) - (1+(length(t)/(step_size_fs) - window_width_fs/step_size_fs)))/2);



%%
%Begin

xx=1;

while xx <= 1+(length(t)/(step_size_fs) - window_width_fs/step_size_fs);

    t_window = [1 + step_size_fs*(xx-1), window_width_fs + step_size_fs*(xx-1)]; %Define time window

    [C,lags] = xcorr(Trace_p(t_window(1):t_window(2)), Trace_d(t_window(1):t_window(2)), range_lags*fs); %calculate xcorr for specific time window, lags are limited to +/- 3ms

    Cross_correlations (:,xx+ajustmnt) = C'; %insert xcorr of this time window into array Cross_correlations. Note the adjustment is done so that the "lost" windows (due to there not being enough data to fill window size) are equally at either side of plot.
    
    xx=xx+1;

end


%%
%Create heatmap plot

y_axis = (-range_lags : 1/fs : range_lags)*1000;
x_axis = (0 : step_size : (width(Cross_correlations)-1)*step_size);

figure;
s=pcolor(x_axis, y_axis, Cross_correlations);
set(s, 'EdgeColor', 'none');
if interp == 1;
    s.FaceColor = 'interp'; 
end
xlabel('time (s)'); 
ylabel('lead/lag (ms)');
set(gcf,'position',[20 450 1250 250])
colormap('bone');

%Create custom color maps to highlight suprathreshold peaks
% Define the number of color levels for each range
nBone = 96;  % Number of grayscale levels
nHot = 64;   % Number of hot levels

% Create individual colormaps
cmapBone = bone(nBone);   % Grayscale colormap for bottom 2/3
cmapHot = hot(nHot);      % Hot colormap for top third

customColormap = [cmapBone(1:64,:); cmapHot(33:64,:)]; % Combine the colormaps (extreme halves of both used only for better combination)
colormap(customColormap); % Apply the custom colormap



if plot_type == 1;
    set(gca, 'yscale', 'log');
    ylabel('lead (ms)');

    figure;
    s2=pcolor(x_axis, -y_axis, Cross_correlations);
    set(s2, 'EdgeColor', 'none');
    if interp == 1;
        s2.FaceColor = 'interp'; 
    end
    xlabel('time (s)'); 
    ylabel('lag (ms)');
    set(gcf,'position',[20 150 1250 250])
    set(gca, 'yscale', 'log');
    colormap('bone');
end

%%
%Calculate X*SD threshold above which peaks are considered significant.

corr_threshold = 2*min(std(Cross_correlations)); %X*SD threshold, min works best, but nanmedian can also work

midpoint_corr = length(Cross_correlations)/2;
color_max = max(Cross_correlations(:,round(midpoint_corr)))*2; %Used to set a guessed top of the color scale
color_min = corr_threshold - 2*(color_max - corr_threshold); %It's important to ensure the 2/3 point of the color scale (the color scheme transition) occurs on the threshold

if color_max > corr_threshold %only apply changed color if range as calculated would include the 4*SD threshold
    clim([color_min color_max]); %Change color scale
else
    disp('threshold outside of color range, range not changed');
end

colorbar; %Add scale bar
%If you want to change clim manually use: 
% color_max = XXX; color_min = corr_threshold - 2*(color_max - corr_threshold);clim([color_min color_max]);

%%
%Create velocity graph corresponding to the plotted lead/lag
%Note you can (and probably want to) change the colourmap limits using clim

dist = 2; %Distance between electrode pair (mm)

v = dist./y_axis;

figure
p=plot(v,y_axis);
set(gcf,'position',[20 460 150 240]);
xlabel('velocity (m/s)'); 
ylabel('lead/lag (ms)');

if plot_type == 1;
    set(gca, 'yscale', 'log');
    ylabel('lead/lag (ms)');

    figure
    p2=plot(v,y_axis);
    set(gcf,'position',[20 160 150 240]);
    xlabel('velocity (m/s)'); 
    ylabel('lag (ms)');
    set(gca, 'yscale', 'log');
end

%%

if sing_plot == 1;

    xx=1;
    
    while xx <= length(t_singleplots);
        
        t_windowsingles = [(t_singleplots(xx)*fs)-(window_width_fs/2) , (t_singleplots(xx)*fs)+(window_width_fs/2)];

        [C,lags] = xcorr(Trace_p(t_windowsingles(1):t_windowsingles(2)), Trace_d(t_windowsingles(1):t_windowsingles(2)), range_lags*fs); %calculate xcorr for specific time window, lags are limited to +/- 3ms
    
        figure 

        plot(y_axis,C);
        hold on
        plot(y_axis(1:61),C(1:61)); %Note this will have to be adjusted if xcorr range is changed
        xlabel('lead/lag (ms)');
        ylabel('Correlation coefficient');

        xx = xx+1;

    end

end








