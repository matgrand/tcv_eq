clear all; close all; clc;


SHOT = 79999; % 83995 % 85508
IP_THRSH = 10000; % I plasma threshold to filter time
mdsopen('tcv_shot', SHOT); % Open the MDSplus connection to the TCV database

fprintf('SHOT number: %d\n', SHOT);


% Call tcvget to retrieve the plasma current (IP)
% We omit the 'time' argument to get the full time trace.
[t_ip, ip_data] = tcvget('IP'); % calculated using magnetics
% [t_ip, ip_data] = tcvget('IPLIUQE'); % calculated using liuqe

% Filter the time vector to remove values below the threshold
good_ip_idxs = find(abs(ip_data) > IP_THRSH);
good_ip_idxs = good_ip_idxs(1:100:end); % decimate the idxs for now 

ts = t_ip(good_ip_idxs);
ips = ip_data(good_ip_idxs);

filtered_percentage = (1 - numel(good_ip_idxs) / numel(ip_data)) * 100;
remaining_samples = numel(good_ip_idxs);
fprintf('Filtered -> %.2f%%, left -> %d samples\n', filtered_percentage, remaining_samples); 

% find the mean and std of the time step
dts = diff(t_ip(good_ip_idxs));
dt_mean = mean(dts);
dt_std = std(dts);
fprintf('time step ->  mean: %.2f µs, std: %.2f µs\n', dt_mean * 1e6, dt_std * 1e6);

% calculate liuqe equilibrium at the good plasma current time
[L,LX,LY] = liuqe(SHOT, t_ip(good_ip_idxs));
Fx = LY.Fx; % Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
Iy = LY.Iy; % Plasma current density map | `(ry,zy,t)` | `[A/m^2]` |
Ia = LY.Ia; % Fitted poloidal field coil currents | `(*,t)` | `[A]` |

Bm = LY.Bm; % Simulated magnetic probe measurements | `(*,t)` | `[T]` |
Ff = LY.Ff; % Simulated flux loop poloidal flux | `(*,t)` | `[Wb]` |

fprintf('Fx -> %s\nIy -> %s\nIa -> %s\nBm -> %s\nFf -> %s\n', ...
    mat2str(size(Fx)), mat2str(size(Iy)), mat2str(size(Ia)), mat2str(size(Bm)), mat2str(size(Ff)));

mdsclose; % Close the MDSplus connection


% save data into a .mat file
save(['data/shot_' num2str(SHOT) '_eq.mat'], 'ts', 'ips', 'Fx', 'Iy', 'Ia', 'Bm', 'Ff');


% Create a 3x2 subplot
plot_idx = round(numel(good_ip_idxs) / 2);
figure

% Subplot 1: Plasma current
subplot(3, 2, 1)
plot(t_ip, ip_data)
hold on
plot(t_ip(good_ip_idxs), ip_data(good_ip_idxs), 'r', 'LineWidth', 2)
line([t_ip(good_ip_idxs(plot_idx)), t_ip(good_ip_idxs(plot_idx))], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5); % Add vertical line
xlabel('Time (s)')
ylabel('Plasma current (A)')
title(['Plasma current for SHOT ' num2str(SHOT)])
grid on

% Subplot 2: Bar plot of Ia at plot_idx
subplot(3, 2, 2)
bar(Ia(:, plot_idx))
xlabel('Coil index')
ylabel('Current (A)')
title('Poloidal field coil currents (Ia)')
grid on

% Subplot 3: Heatmap of Fx at plot_idx
subplot(3, 2, 3)
imagesc(Fx(:, :, plot_idx))
colorbar
xlabel('R index')
ylabel('Z index')
title('Poloidal flux map (Fx)')
axis equal tight

% Subplot 4: Heatmap of Iy at plot_idx
subplot(3, 2, 4)
imagesc(Iy(:, :, plot_idx))
colorbar
xlabel('R index')
ylabel('Z index')
title('Current density map (Iy)')
axis equal tight

% Subplot 5: Bar plot of Bm at plot_idx
subplot(3, 2, 5)
bar(Bm(:, plot_idx))
xlabel('Probe index')
ylabel('Magnetic field (T)')
title('Simulated magnetic probe measurements (Bm)')
grid on

% Subplot 6: Bar plot of Ff at plot_idx
subplot(3, 2, 6)
bar(Ff(:, plot_idx))
xlabel('Flux loop index')
ylabel('Flux (Wb)')
title('Simulated flux loop poloidal flux (Ff)')
grid on

% Save the figure as an SVG file
saveas(gcf, ['figs/shot_' num2str(SHOT) '_eq.svg'])
