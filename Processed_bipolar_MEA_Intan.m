function [Processed,h]=Processed_bipolar_MEA_Intan(amplifier_data,Channels_main,Channels_ref,Timepoints)

%Processes intan samples by setting up bipolars between one or more
%electrodes (Set in a column vector Channels_main) and an equivalent number
%of reference electrodes (Set in a matrix Channels_ref). The format of Channels_ref is different.
%It must contain 32 elements per row, with '1' for channels used and '0'
%for channels not used (when setting reference channel this will be done by
%multiplicating arrays). Build in excel for sanity.
%E.g. Channels_main = [1;4;7];
%       Channels_ref = [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
% 0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
% 0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
%
%Timepoints adds vertical lines to plots based on provided timepoints (in
%seconds). Expects a vector e.g. [2, 3.5, 6.242];

%% 
%Begin processing by contructing variables with channels 

A=amplifier_data;

n=numel(Channels_main); %Number of channels to analyse
n2=sum(not(isnan(Channels_ref)),2); %Number of channels in each reference (vector, ignores NaNs)

clear Imported_electrodes;  %Otherwise if this already exists it may produce an error in while loop below
clear Filtered_electrodes; 

DS=1; %Downsampling factor
A=downsample(A',DS);%Downsample
A=A';

%%
%Generate filter for signals. By default uses bandpass butterworth 300-2000Hz, 4th
%order. Also plots filter transfer function

fc = [300 2000];
fs = 30000/DS;

[b,a] = butter(4,fc/(fs/2),'bandpass');

Fig = figure('Name','Filter transfer function');

freqz(b,a,[],fs)

subplot(2,1,1);
ylim([-100 20])

%%
%Reference and filter signals

xx=1;
while (xx<=n)
    Reference_matrix = A.*repmat(Channels_ref(xx,:)',[1 length(A)]);
    Reference_vector = sum(Reference_matrix,1);
    Reference_vector_mean = Reference_vector/sum(sum(Channels_ref(xx,:)));  %Multiplies channels by Channels_ref (a particular row for each loop) to obtain data for referencing, and divide to obtain mean
    Imported_electrode = A(Channels_main(xx),:);        %Select channel to process as main channel (from Channels_main)
    assignin('base',strcat('Imported_electrode',num2str(xx)),Imported_electrode-Reference_vector_mean); %Generate a new variable from the mean
    Imported_electrodes(xx,:) = Imported_electrode-Reference_vector_mean;  %Output the whole thing as a single matrix as well
    Filtered_electrodes(xx,:) = filter(b,a,Imported_electrode-Reference_vector_mean); %Carry out filtering of referenced electrode and build whole filtered signal as output
    xx=xx+1;
end



%% 
%Plot figures
%Ylim can also be changed here (or a forced value removed altogether)

L = length(A); %Used to remap X axis in figures to seconds
Lsmol= L/fs; 

Timepoints_datapoints=Timepoints*fs; %Converts datapoints back from seconds to datapoint
Timepoints_datapoints(2,:)=Timepoints_datapoints; %Prepares array to add vertical lines for timepoints

Fig3 = figure('Name','Referenced data');
xx=1;
while (xx<=n)
    h=subplot(n,1,xx);
    plot(Imported_electrodes(xx,:));
    set(gca,'XTick',0:round(Lsmol/10)*fs:L) %Generates ticks based on duration of recording (samples). Determined from seconds to have nice round numbers after transforming back
    set(gca,'XTickLabel',0:round(Lsmol/10):Lsmol) %Shows ticks as reconverted to time
    ylabel('Voltage (uV)')
    ylim([-2500 2500]); %Set ylim between -250 and 250 uA    
    line(Timepoints_datapoints,[-10000 10000],'Color','red','LineStyle','--') 
    xticks([]);
    yticks([]);
    ylabel(join(['Ch',string(xx)]));
    xx=xx+1;
end
xlabel('time (s)')
linkaxes

%%Calculate RMS and perform rolling window of specified length and plot it
%Median value is subtracted to remove influence of noise on value

Window_length = 0.02; %Mean window length, in seconds.
Fig4 = figure('Name','RMS Mean window');
xx=1;
while (xx<=n)
    h=subplot(n,1,xx);
    Processed_RMSWind=movmean(rms(Filtered_electrodes(xx,:),1),Window_length*fs);
    plot(Processed_RMSWind-median(Processed_RMSWind));
    set(gca,'XTick',0:round(Lsmol/10)*fs:L)
    set(gca,'XTickLabel',0:round(Lsmol/10):Lsmol)
    ylabel('Voltage (uV)')
    ylim([-3 30]); %Set ylim between 0 and 25 uA
    line(Timepoints_datapoints,[-10000 10000],'Color','red','LineStyle','--')
    xticks([]);
    yticks([]);
    ylabel(join(['Ch',string(xx)]));
    xx=xx+1;
end
linkaxes;
xlabel('time (s)')

Fig = figure('Name','Filtered data');
xx=1;
while (xx<=n)
    h=subplot(n,1,xx);
    plot(Filtered_electrodes(xx,:));
    set(gca,'XTick',0:round(Lsmol/10)*fs:L)
    set(gca,'XTickLabel',0:round(Lsmol/10):Lsmol)
    ylabel('Voltage (uV)')
    ylim([-50 50]); %Set ylim between -25 and 25 uA
    line(Timepoints_datapoints,[-10000 10000],'Color','red','LineStyle','--')
    xticks([]);
    yticks([]);
    ylabel(join(['Ch',string(xx)]));
    xx=xx+1;
end
linkaxes;
xlabel('time (s)')

%%
%Generate outputs

Processed = Filtered_electrodes;
h = Fig;









