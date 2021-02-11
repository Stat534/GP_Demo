data {
  int<lower=0> N; // number of data points
  vector[N] y; // responds
  matrix[N,N] dist; // distance matrix
}

parameters {
  real<lower = 0.5, upper = 9.8> phi;
  real<lower = 0> sigmasq;
  real mu;
}

transformed parameters{
  vector[N] mu_vec;
  corr_matrix[N] Sigma;
  
  for(i in 1:N) mu_vec[i] = mu;
  for(i in 1:(N-1)){
   for(j in (i+1):N){
     Sigma[i,j] = exp((-1)*dist[i,j]/ phi);
     Sigma[j,i] = Sigma[i,j];
   }
 }
 for(i in 1:N) Sigma[i,i] = 1;

}

model {
  y ~ multi_normal(mu_vec ,sigmasq * Sigma);
  phi ~ inv_gamma(10,10);
  sigmasq ~ inv_gamma(10,10);
  mu ~ normal(0, 10);
}

