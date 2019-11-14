data {

  int<lower=0> N;             // number of data points
  vector<lower=0>[N] x_obs;            // x observations
  vector<lower=0>[N] y_obs;            // y observations  
  vector<lower=0>[N] sigma_x; // measurement uncertainty in x
  vector<lower=0>[N] sigma_y; // measurement uncertainty in y
  vector<lower=0>[N] dist_obs;         // distance to objects

  vector[N] family_idx;       // identifier if object is diffuse (0) or GC (1) or satellite (2)
  
  int<lower=0> N_model;       // number of point to evaluate the fitted model
  vector<lower=0>[N_model] x_model;    // where to evaluate the model

  //priors on the fitted parameters
  real mu_lgNorm;
  real sigma_lgNorm;
  real mu_alpha;
  real sigma_alpha;

  // fluxes (y) are much better determined than masses (x)
  // and GCs might have different systematics than large diffuse components
  real sys_xy_width_diffuse;
  real sys_xy_width_GC;
  
  // masses of all GC to predcit the 511 keV flux
  int<lower=0> NGC;  // number of objects to predict flux
  vector<lower=0>[NGC] GC_masses;
  vector<lower=0>[NGC] GC_masses_err;
  vector<lower=0>[NGC] GC_dist;
  
}

transformed data {
 
    vector[NGC] lgGC_masses = log10(GC_masses);
    vector[NGC] lgGC_masses_err;
    for (ngc in 1:NGC) {
        lgGC_masses_err[ngc] = (fabs(log10(GC_masses[ngc]) - log10(fabs(GC_masses[ngc]-GC_masses_err[ngc]))) + fabs(log10(fabs(GC_masses[ngc]+GC_masses_err[ngc])) - log10(GC_masses[ngc])))/2.;
    }
    
}

parameters {
  
  real lgNorm;  // normalisation of power-law function
  real alpha;   // power-law index
  vector[N] lgx_latent; // latent x location (x_obs + randomn(sigma_x))
  real<lower=1> sys_xy_diffuse;   // scale for "systematic uncertainty" in logx and logy for diffuse components
  real<lower=1> sys_xy_GC;        // scale ... logx and logy ... GC
}

transformed parameters {

  // observed lgy values
  vector[N] lgy_obs = log10(y_obs);
  // latent lgy values, not obscured by measurement error
  vector[N] lgy_true = lgNorm + lgx_latent*alpha;
  // y and x uncertainty in dex
  vector[N] lgsigma_y; 
  vector[N] lgsigma_x;
  // and systematics
  vector[N] lgsys_y;
  vector[N] lgsys_x;
  // observed lgx values
  vector[N] lgx_obs = log10(x_obs);
  // estimate for y and x uncertainty in dex
  for (n in 1:N) {
      lgsigma_y[n] = (fabs(log10(y_obs[n]) - log10(fabs(y_obs[n]-sigma_y[n]))) + fabs(log10(fabs(y_obs[n]+sigma_y[n])) - log10(y_obs[n])))/2.;
      lgsigma_x[n] = (fabs(log10(x_obs[n]) - log10(fabs(x_obs[n]-sigma_x[n]))) + fabs(log10(fabs(x_obs[n]+sigma_x[n])) - log10(x_obs[n])))/2.;
      
  // systematics-affected uncertainty
      if (family_idx[n] == 0) {
          lgsys_y[n] = lgsigma_y[n]*sys_xy_diffuse;
          lgsys_x[n] = lgsigma_x[n]*sys_xy_diffuse;
      }
      else if (family_idx[n] == 1) {
          lgsys_y[n] = lgsigma_y[n]*sys_xy_GC;
          lgsys_x[n] = lgsigma_x[n]*sys_xy_GC;
      }
      else {
          lgsys_y[n] = lgsigma_y[n]*sys_xy_diffuse;
          lgsys_x[n] = lgsigma_x[n]*sys_xy_diffuse;
      }
  }
}

model {

  // priors
  lgNorm ~ normal(mu_lgNorm,sigma_lgNorm);
  alpha ~ normal(mu_alpha,sigma_alpha);
  sys_xy_diffuse ~ cauchy(1.,sys_xy_width_diffuse);
  sys_xy_GC ~ cauchy(1.,sys_xy_width_GC);
  
  lgx_latent ~ normal(lgx_obs,lgsys_x);

  // likelihood

  lgy_obs ~ normal(lgy_true, lgsys_y);
  
 
}

generated quantities {

  vector<lower=0>[N] ppc_x;
  vector<lower=0>[N] ppc_y;
  
  vector<lower=0>[NGC] GC_ppc_x;
  
  vector[N_model] powerlaw;
  vector[N_model] powerlaw511;
    
  vector<lower=0>[N] five11_flux_sample;
  real<lower=0> GC511_flux_total_sample = 0.;

  vector<lower=0>[NGC] five11_flux_GC;
  real<lower=0> GC511_flux_total_GC = 0.;
  
  // generate posteriors
  // fitted powerlaw
  for (nm in 1:N_model) {
      powerlaw[nm] = pow(10.,lgNorm) * pow(x_model[nm],alpha);
      powerlaw511[nm] = powerlaw[nm]*normal_rng(0.32,0.03); // in erg/s @ 511 keV
  }

  // create posterior samples for PPC
  for (n in 1:N) {

    ppc_x[n] = pow(10.,normal_rng(lgx_latent[n], lgsys_x[n]));
    ppc_y[n] = pow(10.,normal_rng(lgNorm + lgx_latent[n]*alpha, lgsys_y[n]));
    
    // prediction of 511 keV flux assuming conversion factor bewteen GeV and 511 keV as (1.5+-0.1)e-3 GeV/ph (Bartels+2018)
    // or in luminosity in the bulge: 2e37 erg/s (+-10%) a 0.96e-3 ph/cm2/s (+-5%) == 6.3e36 erg/s --> (0.32 +- 0.03) erg/s / (erg/s)
    // 4 pi kpc**2 == 1.1965e44 cm**2
    // 1 erg == 1221430 (511 keV) photons
    five11_flux_sample[n] = ppc_y[n]*normal_rng(0.32,0.03)/(dist_obs[n]^2*1.1965e44)*1.22143e6;
    if (family_idx[n] == 1) {
        GC511_flux_total_sample += five11_flux_sample[n];
    }
      
  }
  
  // create predictions for all GCs
  for (ngc in 1:NGC) {
    
    GC_ppc_x[ngc] = pow(10.,normal_rng(lgGC_masses[ngc], lgGC_masses_err[ngc]*sys_xy_GC));  
      
    // prediction of 511 keV flux assuming conversion factor bewteen GeV and 511 keV as (1.5+-0.1)e-3 GeV/ph (Bartels+2018)
    // or in luminosity in the bulge: 2e37 erg/s (+-10%) a 0.96e-3 ph/cm2/s (+-5%) == 6.3e36 erg/s --> (0.32 +- 0.03) erg/s / (erg/s)
    five11_flux_GC[ngc] = pow(10.,(lgNorm + normal_rng(lgGC_masses[ngc], lgGC_masses_err[ngc]*sys_xy_GC)*alpha))*normal_rng(0.32,0.03)/(GC_dist[ngc]^2*1.1965e44)*1.22143e6;
    GC511_flux_total_GC += five11_flux_GC[ngc];
    
  }
  
  

}