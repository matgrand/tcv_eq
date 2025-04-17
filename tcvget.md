Okay, here is a comprehensive Markdown documentation file for the `tcvget` function, packed with examples covering its main features.

```markdown
# `tcvget` Function Documentation (TCV MATLAB)

## 1. Overview

`tcvget` is a fundamental MATLAB function within the TCV data analysis environment. Its primary purpose is to retrieve a wide variety of physics parameters, diagnostic signals, calculated quantities, and metadata associated with a specific TCV plasma discharge (shot) from the MDSplus database.

It acts as a unified interface, simplifying data access by mapping human-readable signal names (like `'IP'` for plasma current) to the underlying MDSplus data nodes and performing necessary calculations or interpolations.

## 2. Prerequisites

*   **MATLAB:** The function is designed to be run within a MATLAB environment.
*   **TCV MDSplus Environment:** Your MATLAB session must be configured to access the TCV MDSplus trees.
*   **Shot Open:** Crucially, you **must** open the desired TCV shot *before* calling `tcvget` using the `mdsopen` command.

```matlab
% --- !!! ALWAYS DO THIS FIRST !!! ---
shot = 83995; % Replace with your desired shot number
mdsopen('tcv_shot', shot); % Replace 'tcv_shot' if your TCV tree name is different
% ------------------------------------
```

## 3. Syntax

```matlab
[t, x] = tcvget(signal_name [, time_vector] [, ParamName, ParamValue, ...]);
```

**Inputs:**

1.  `signal_name` (String, **Required**):
    *   The specific signal you want to retrieve. This is **case-insensitive**.
    *   Must exactly match one of the many keywords listed in the `help tcvget` output (e.g., `'IP'`, `'NEL'`, `'WMHD'`, `'TE0'`, `'RGEO'`, `'TIMEDISR'`).
    *   Refer to `help tcvget` for the definitive list and a brief description of each signal.

2.  `time_vector` (Numeric Vector, Optional):
    *   A vector specifying the exact time points (in seconds) at which you want the signal's value.
    *   If provided, `tcvget` will typically interpolate or select the data closest to these times.
    *   If omitted or empty (`[]`), the function usually returns the signal's complete time trace as stored in the database or calculated.

3.  `ParamName, ParamValue` (String/Value Pairs, Optional):
    *   Additional arguments passed as name-value pairs to control specific behaviours, especially for calculated or equilibrium-dependent signals. Key parameters include:
        *   `'source'`: (String) Specifies the data source, primarily for equilibrium quantities (LIUQE reconstructions). Common values seen in the code:
            *   `'LIUQE.m'` (often the default or recommended)
            *   `'LIUQE'`
            *   `'LIUQE2'`, `'LIUQE3'`
            *   `'FBTE'`
            *   *Effect:* Determines which specific equilibrium reconstruction results are used for signals like `AMIN`, `RGEO`, `WMHD`, `Q95`, `KAPPA`, `DELTA`, etc. Check the `help tcvget` section for which signals support this.
        *   `'tcv_eq_keyword'`: (String) Keyword passed directly to underlying `tcv_eq` calls (advanced use).
        *   `'dyy_crit'`, `'dtelm_min'`: (Numeric) Threshold parameters used by the ELM detection algorithm when retrieving signals like `TIMOFELM`.

**Outputs:**

*   `t` (Numeric Vector or `NaN`):
    *   The time vector (column vector) corresponding to the data points in `x`.
    *   Units are typically seconds.
    *   If the requested signal is a constant (scalar) and no `time_vector` was provided, `t` will be `NaN`.
    *   If a `time_vector` was provided, `t` will usually be identical to the input `time_vector`.
*   `x` (Numeric Vector, String, or `NaN`):
    *   The data vector (column vector) containing the signal values at the times specified in `t`.
    *   Units depend on the specific `signal_name` (e.g., Amperes for `IP`, m⁻³ for `NEL`, Joules for `WMHD`, eV for `TE0`). Refer to physics context or potentially other documentation for units.
    *   For certain signals (like `'PDJ'`, `'SL'`, `'SHOTDATE'`), `x` will be a character array (string).
    *   If data is unavailable, the calculation fails, or interpolation is not possible, `x` may contain `NaN` values.

## 4. Core Functionality & Implementation

*   **MDSplus Interaction:** `tcvget` heavily relies on underlying MDSplus functions (`mdsopen`, `mdsvalue`, `mdsdata`, potentially others like `tdi`) to read data from the TCV trees.
*   **Signal Mapping:** A large `switch` statement maps the input `signal_name` to the appropriate actions:
    *   Direct node access (e.g., `mdsvalue('\results::fir:n_average')` for `'NEL'`).
    *   Calling specific TCV functions (e.g., `tcv_ip()` for `'IP'`, `ts_fitdata(...)` for Thomson profiles, `tcv_eq(...)` for equilibrium data).
    *   Performing calculations (e.g., derivatives, sums, ratios of other retrieved signals).
*   **Equilibrium Dependence:** Many geometry and MHD parameters (`RGEO`, `AMIN`, `WMHD`, `Q95`, `KAPPA`, `DELTA`, etc.) depend on prior equilibrium reconstructions (LIUQE). The `'source'` parameter selects which reconstruction results to use.
*   **Interpolation/Sampling:** When a `time_vector` is supplied, internal routines (`dbsamp`, `interp1`, `interpos`) are used to provide data at the requested times, often using nearest-neighbor, linear, or spline interpolation based on the signal type and internal logic.
*   **Helper Functions:** May call other specialized TCV MATLAB functions (e.g., `stana` for sawtooth analysis, `get_gas_flux` for gas puffing).

## 5. Key Features & Examples

**(Remember to run `mdsopen` before these examples!)**

```matlab
% --- Setup for Examples ---
shot = 83995; % Use a recent, potentially well-diagnosed shot
mdsopen('tcv_shot', shot);
% --------------------------
```

**5.1. Basic Signal Retrieval (Full Time Trace)**

*   Retrieve the entire time history of a standard signal.

```matlab
% Get Plasma Current (IP)
[t_ip, ip_data] = tcvget('IP'); % IP is in Amperes

% Get Line-Averaged Density (Central FIR chord, default)
[t_nel, nel_data] = tcvget('NEL'); % NEL is in m^-3

% Get Stored Energy (from default equilibrium source)
[t_wmhd, wmhd_data] = tcvget('WMHD'); % WMHD is in Joules

% Plot IP
figure;
plot(t_ip, ip_data / 1e3); % Convert A to kA for plotting
xlabel('Time (s)');
ylabel('Plasma Current (kA)');
title(['IP for Shot #', num2str(shot)]);
grid on;
```

**5.2. Retrieval at Specific Times**

*   Get signal values only at the time points you specify.

```matlab
my_times = [0.5, 0.8, 1.0, 1.2, 1.5]'; % Define desired times (column vector)

% Get Plasma Current at specific times
[t_ip_pts, ip_pts] = tcvget('IP', my_times);

% Get Central Electron Temperature (Thomson Scattering) at specific times
[t_te0_pts, te0_pts] = tcvget('TE0', my_times); % TE0 is in eV

% Display results
disp('IP at specified times:');
disp([t_ip_pts, ip_pts / 1e3]); % Show time and IP (kA)

disp('TE0 at specified times:');
disp([t_te0_pts, te0_pts]); % Show time and TE0 (eV)

% Note: t_ip_pts should be identical to my_times if interpolation succeeds.
```

**5.3. Using Optional Parameters (`source`)**

*   Specify the source for equilibrium-dependent signals.

```matlab
% Get minor radius using the default source (likely LIUQE.m)
[t_a_def, a_def] = tcvget('AMIN');

% Get minor radius explicitly using LIUQE.m
[t_a_liuqem, a_liuqem] = tcvget('AMIN', [], 'source', 'LIUQE.m');

% Get minor radius using the older LIUQE tree structure (if available)
try % Use try-catch as older sources might not exist for all shots
    [t_a_liuqe, a_liuqe] = tcvget('AMIN', [], 'source', 'LIUQE');
    disp('Successfully retrieved AMIN using source LIUQE');
catch ME
    warning('Could not retrieve AMIN using source LIUQE: %s', ME.message);
end

% Compare default and explicit LIUQE.m (should be very similar or identical)
figure;
plot(t_a_def, a_def, 'b-', 'LineWidth', 1.5);
hold on;
plot(t_a_liuqem, a_liuqem, 'r--');
% if exist('t_a_liuqe', 'var')
%     plot(t_a_liuqe, a_liuqe, 'g:');
%     legend('Default Source', 'Source LIUQE.m', 'Source LIUQE');
% else
    legend('Default Source', 'Source LIUQE.m');
% end
xlabel('Time (s)');
ylabel('Minor Radius AMIN (m)');
title(['AMIN Comparison for Shot #', num2str(shot)]);
grid on;
```

**5.4. Calculated Signals (Derivatives, Combinations)**

*   Retrieve signals calculated on-the-fly by `tcvget`.

```matlab
% Get time derivative of plasma current
[t_dipdt, dipdt_data] = tcvget('DIPDT'); % Units: A/s

% Get Total Injected Power (Ohmic + ECH + NBI)
% This might require NBH/NB2/DNBI and TORAY analysis to be present
[t_ptot, ptot_data] = tcvget('PTOT'); % Units: Watts

% Get Normalized Beta (depends on BETMHD, IP, AMIN, BT)
[t_ben, ben_data] = tcvget('BENMHD'); % Dimensionless (%*m*T/MA)

% Plot dIp/dt
figure;
plot(t_dipdt, dipdt_data / 1e6); % Convert A/s to MA/s
xlabel('Time (s)');
ylabel('dIP/dt (MA/s)');
title(['Rate of Change of IP for Shot #', num2str(shot)]);
grid on;

% Plot Ptot
figure;
plot(t_ptot, ptot_data / 1e6); % Convert W to MW
xlabel('Time (s)');
ylabel('Total Power (MW)');
title(['P_{TOT} for Shot #', num2str(shot)]);
grid on;
```

**5.5. Geometry & Equilibrium Signals**

*   Retrieve key plasma shape and MHD parameters.

```matlab
% Get Geometrical Major Radius
[t_rgeo, rgeo_data] = tcvget('RGEO'); % Units: m

% Get Safety Factor at 95% flux surface
[t_q95, q95_data] = tcvget('Q95'); % Dimensionless

% Get Elongation at the edge
[t_kappa, kappa_data] = tcvget('KAPPA'); % Dimensionless

% Get Upper Triangularity at 95% flux surface
[t_dup95, dup95_data] = tcvget('DELTAU_95'); % Dimensionless

% Plot Q95
figure;
plot(t_q95, q95_data);
xlabel('Time (s)');
ylabel('Safety Factor q_{95}');
title(['q_{95} for Shot #', num2str(shot)]);
ylim([0, 10]); % Adjust Y limits if needed
grid on;
```

**5.6. Profile Data (Point Values)**

*   Retrieve point values from fitted profiles (e.g., Thomson Scattering).

```matlab
% Get Central Electron Temperature (Te at rho=0)
[t_te0, te0_data] = tcvget('TE0'); % Units: eV

% Get Electron Density at rho=0.9 (near edge)
[t_ne90, ne90_data] = tcvget('NE90'); % Units: m^-3

% Plot Te0
figure;
plot(t_te0, te0_data / 1e3); % Convert eV to keV
xlabel('Time (s)');
ylabel('Central Electron Temperature T_{e0} (keV)');
title(['T_{e0} for Shot #', num2str(shot)]);
grid on;
```

**5.7. Event Times**

*   Retrieve timings of specific events like disruptions or ELMs.

```matlab
% Get Disruption Time (if any)
[t_disr_req, time_disrupt] = tcvget('TIMEDISR');
% Note: time_disrupt will be a scalar value repeated for all requested times,
% or a single scalar if no time vector was given. t_disr_req matches input time.
if ~isnan(time_disrupt(1))
    fprintf('Shot %d disrupted at t = %.4f s\n', shot, time_disrupt(1));
else
    fprintf('Shot %d did not have a logged disruption time.\n', shot);
end

% Get times of ELM events (requires D-alpha analysis)
% This returns a list of times when ELMs occurred.
[t_elms, x_elms] = tcvget('TIMOFELM');
% Here, t_elms and x_elms contain the *same* list of ELM times.
if ~isempty(t_elms) && ~ischar(t_elms) && ~all(isnan(t_elms))
    fprintf('Found %d ELMs for shot %d. First ELM at %.4f s, Last ELM at %.4f s\n', ...
            length(t_elms), shot, t_elms(1), t_elms(end));
    % Example: Get Q95 just before the first few ELMs
    if length(t_elms) >= 3
        elm_query_times = t_elms(1:3) - 0.001; % Query 1ms before ELM crash
        [t_q95_elms, q95_at_elms] = tcvget('Q95', elm_query_times);
        disp('Q95 just before first 3 ELMs:');
        disp([elm_query_times, q95_at_elms]);
    end
else
    fprintf('No ELMs detected or analyzable for shot %d.\n', shot);
end
```

**5.8. Handling Constants/Scalars**

*   Some signals represent constant values for the shot.

```matlab
% Get Vacuum Toroidal Field at nominal major radius (R=0.88m)
% Request without time vector:
[t_b0, b0_val] = tcvget('B0');
fprintf('B0 for shot %d: %.3f T (t output is NaN: %d)\n', shot, b0_val, isnan(t_b0));

% Request B0 at specific times:
my_times = [0.5, 1.0, 1.5]';
[t_b0_pts, b0_pts] = tcvget('B0', my_times);
disp('B0 requested at specific times:');
disp([t_b0_pts, b0_pts]); % Note: b0_pts repeats the constant value

% Get Shot Date (string constant)
[t_date, date_str] = tcvget('SHOTDATE');
fprintf('Shot Date for shot %d: %s (t output is NaN: %d)\n', shot, date_str, isnan(t_date));
```

**5.9. Handling String Outputs**

*   Retrieve metadata stored as strings.

```matlab
% Get username of the Physics Operator (PDJ)
[t_pdj, pdj_name] = tcvget('PDJ'); % t_pdj should be NaN
if ~isempty(pdj_name)
    fprintf('PDJ for shot %d: %s\n', shot, pdj_name);
else
    fprintf('PDJ not found for shot %d.\n', shot);
end

% Get username of the Session Leader (SL)
[t_sl, sl_name] = tcvget('SL'); % t_sl should be NaN
if ~isempty(sl_name)
    fprintf('SL for shot %d: %s\n', shot, sl_name);
else
    fprintf('SL not found for shot %d.\n', shot);
end
```

```matlab
% --- Cleanup ---
mdsclose;
% ---------------
```

## 6. Important Considerations & Best Practices

*   **`mdsopen` is Mandatory:** Always ensure the correct shot is open before calling `tcvget`. Forgetting this is a common source of errors or unexpected results (retrieving data from the wrong shot).
*   **Check `help tcvget`:** This is the definitive source for the available `signal_name` keywords and often provides hints about units or calculation methods. The list is extensive.
*   **Signal Availability:** Not all signals are available for all shots. Diagnostics may not have been operational, or specific analyses (like equilibrium reconstructions or profile fitting) may not have been performed or stored successfully. Expect to encounter `NaN` values in the output `x`.
*   **Performance:** Retrieving directly measured signals is usually fast. Retrieving calculated signals,w especially those requiring complex analysis or reading large profile datasets, can take significantly longer.
*   **Units:** `tcvget` itself doesn't explicitly return units. You need to know the expected units from the context of the signal name (e.g., Amps for `IP`, meters for `RGEO`, eV for `TE0`).
*   **Equilibrium Source (`'source'`)**: Be mindful of the `'source'` parameter when retrieving geometry or MHD parameters. The default (`LIUQE.m`) is often suitable, but results can differ slightly between sources. Consistency might be important for comparisons.
*   **Error Handling:** Wrap calls in `try-catch` blocks if you are scripting analysis over many shots, as some signals might be missing or cause errors for specific discharges. Check if the returned `x` contains only `NaN`s.

## 7. Troubleshooting

*   **Error "MDSplus error ... Tree not open":** You forgot to call `mdsopen` or called `mdsclose` too early.
*   **Output `x` is all `NaN`s:**
    *   The signal genuinely wasn't available for that shot (diagnostic off, analysis not run).
    *   You might have misspelled the `signal_name`. Check `help tcvget`.
    *   For calculated signals, one of the input signals needed for the calculation might be missing.
    *   For equilibrium signals, the specified `'source'` might not exist for that shot. Try the default or `'LIUQE.m'`.
*   **Unexpected Values:** Double-check the `signal_name` and expected units. Ensure you didn't accidentally retrieve data from the wrong shot. Consider the `'source'` if it's an equilibrium quantity.

## 8. Common Signal Names (Quick Reference Subset)

*   `IP`, `IPMA`: Plasma current (A, MA)
*   `NEL`, `NEL19`, `NEL20`: Line-averaged density (m⁻³, 10¹⁹ m⁻³, 10²⁰ m⁻³)
*   `TE0`, `TE95`: Electron temperature (central, rho=0.95) (eV)
*   `WMHD`: Stored energy (MHD) (J)
*   `PTOT`, `PECH`, `POHM`, `PNBI`: Powers (Total, ECH, Ohmic, NBI) (W)
*   `Q95`, `Q0`: Safety factor (rho=0.95, central)
*   `RGEO`, `AMIN`: Major, minor radius (geometric) (m)
*   `KAPPA`, `DELTA`: Elongation, Triangularity
*   `BT`, `B0`: Toroidal field (at Rgeo, at R=0.88m) (T)
*   `PRAD`: Radiated power (W)
*   `DALPHA`: D-alpha emission signal (arbitrary/calibrated units)
*   `TIMEDISR`: Disruption time (s)
*   `TIMOFELM`: Times of ELM occurrences (s)
*   `SHOTDATE`, `SHOTTIME`: Shot date and time (string, scalar hours)
*   `PDJ`, `SL`: Usernames (string)

**(Always consult `help tcvget` for the full, up-to-date list)**

---

*Disclaimer: This documentation is based on the provided help text and implementation code. The exact behavior and available signals might depend on the specific version of the TCV MATLAB tools and MDSplus database structure.*
```