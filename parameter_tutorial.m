% Parameter Estimation Tutorial
% Driver for the parameter estimation tutorial on ***WEBSITE***
%
%    INPUTS:
%        is_discrete: indicator whether you want to estimate using discrete
%            approximations (1) or continuous/spline approximations (0)
%
%    OUTPUTS:
%        precomputed_sol: a mat-file with the full forward solutions over  
%             the parameter space of interest (named 'fullsol')
%
% Written by Erica Rutter (February 2020)

%% create synthetic data do do parameter estimation
t0=0;
tf=30;
y0=10;
true_par=[0.5,200];
ts=linspace(t0,tf,51);
[~,y]=ode45(@logistic,ts,y0,[],true_par);

% create synthetic data with 10% proportional error
sigma=0.1;
prop_data = y.*(1+sigma*rand(size(y)));
save('prop_data.mat','prop_data')

% create synthetic data with constant error
const_data = y+20*rand(size(y));
save('const_data.mat','const_data')

%% OLS Formulation with constant error data
init_guess=[0.2,100];
ols_error_estimate = @(par)ols_formulation(par,const_data,ts,y0);
options = optimset('MaxIter', 25000, 'MaxFunEvals', 50000, 'Display', 'off');
[ols_optpar,~,converge_flag,~]=fminsearch(ols_error_estimate, init_guess,options);
disp('OLS Estimation')

%% GLS Formulation with proportional error data
gammas=1;
weights=ones(size(prop_data));
gls_optpar=init_guess;
old_gls_optpar=init_guess;
tol=1e-4;
maxits=2000;
minits=10;
partol = 0.1;
parchange = 100;
oldparchange = 100;
ii = 1;
while ii<maxits && parchange > partol && oldparchange > partol || ii< minits
    gls_error_estimate = @(par)gls_formulation(par,prop_data,ts,y0,weights);
    options = optimset('MaxIter',2500,'MaxFunEvals',5000,'Display', 'Off');
    gls_optpar=fminsearch(gls_error_estimate, gls_optpar,options);
    [~,weights]=ode45(@logistic,ts,y0,[],gls_optpar); 
    weights(weights<tol)=0;
    weights(weights>tol)=weights(weights>tol).^(-2*gammas);
    inds=old_gls_optpar>1e-10;
    weights=full(weights);
    parchange =1/(2)*sum((abs(gls_optpar(inds)-old_gls_optpar(inds))./old_gls_optpar(inds)));
    ii = ii+1;
    old_gls_optpar=gls_optpar;
end
disp('OLS Estimation')
ols_optpar
disp('GLS Estimation')
gls_optpar

%% Residual Examination

% OLS Solution Residual
[~,ols_sol]=ode45(@logistic,ts,y0,[],ols_optpar);
ols_resids=const_data-ols_sol;

% GLS Solution Residual    
[~,gls_sol]=ode45(@logistic,ts,y0,[],gls_optpar); 
gls_resids=(prop_data-gls_sol)./(gls_sol.^gammas);

%Plot the Residuals versus model value and time!
figure
subplot(2,2,1)
plot(ts,ols_resids,'b*')
set(gca,'Fontsize',18,'linewidth',1.5)
title('OLS')
xlabel('Time (Days)')
ylabel('Residual')
subplot(2,2,2)
plot(ts,gls_resids,'r*')
set(gca,'Fontsize',18,'linewidth',1.5)
title('GLS')
xlabel('Time (Days)')
subplot(2,2,3)
plot(ols_sol,ols_resids,'b*')
set(gca,'Fontsize',18,'linewidth',1.5)
xlabel('Model Value')
ylabel('Residual')
subplot(2,2,4)
plot(gls_sol,gls_resids,'r*')
set(gca,'Fontsize',18,'linewidth',1.5)
xlabel('Model Value')


%% Problem 1. Do the OLS formulation for prop_data and examine residuals


%% Problem 2. Do the GLS formulation for const_data and examine residuals


%% Define the logistic growth curve
% Defines the right-hand side of the logistic equation to be solved in 
function dydt = logistic(t,y,par)
dydt=par(1)*y*(1-y/par(2));
end

%% Define the OLS formulation
function resid = ols_formulation(par,data,ts,y0)
[~,y]=ode45(@logistic,ts,y0,[],par);
resid = sum((data-y).^2);
end

%% Define the GLS formulation
function resid = gls_formulation(par,data,ts,y0,weights)
[~,y]=ode45(@logistic,ts,y0,[],par);
resid = sum(weights.*((data-y).^2));
end
