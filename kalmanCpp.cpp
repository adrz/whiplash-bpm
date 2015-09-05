#include <RcppArmadillo.h>
//#include <stdio.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List findPeaksCpp(NumericVector X, double threshold, double maxBPM, bool norm, double fs)
{
	int N = X.size();
    arma::colvec x(X.begin(), X.size(), true);       // reuses memory

	int it = 0;
	int samples_ignored = 1/maxBPM*fs*60;
	int toignore = 0;

	while (1){
		if (x(it) > threshold){
			if (norm)
			{
				x(it) = 1;
			}
			toignore = std::min(it+samples_ignored,N-1) - it;
			x.subvec(it+1,it+toignore) = zeros<vec>(toignore);
			it+=toignore;
		}
		else{
			x(it) = 0;
			it++;
		}
		if (it == N)
			break;
	}

	
    return Rcpp::List::create(
    Rcpp::Named("x") = x
    ) ;
	
}

// !!! ONLY VALID FOR AR 1
// [[Rcpp::export]]
List kalmanCpp(NumericVector Sig, double Q, double H, double F, 
                    double R, double rho2,
			   NumericVector Qt_, int winnoise, int overlap, float min_sigma2b)
{
    int N = Sig.size();
    arma::colvec sig(Sig.begin(), Sig.size(), false);       // reuses memory 
    arma::colvec Qt(Qt_.begin(), Qt_.size(), false);       // reuses memory and avoids extra copy
    

    double e;
    double S;
    double Sinv;
    double kt;
    double z;
    double st;
    vec st0;

  	double xt_t = 0;
  	double xt_t1 = 0;
  	double xt1_t1 = 0;
	  double pt_t = 0;
  	double pt1_t1 = 0;
  	double pt_t1 = 0;
  	double sigma2b = 1;
  
  	int noiseestim = 0;
  	bool th = 1;
  
    for (uint it = 1; it<N; it++)
    {
    
		if ((it > winnoise) & (noiseestim > overlap) ){
		 	sigma2b = var(sig.subvec(it - winnoise ,it));
			if (th){
				if (sigma2b < min_sigma2b) // .00001
				{
					sigma2b = min_sigma2b;
				}
				if (sigma2b > .1){
				}
			}
		 	noiseestim = 0;
		}
		
		noiseestim++;
		z  = sig(it);
    // ========
    // Predict 
    // ========

    // Update apriori estimate
    xt_t1        = F*xt1_t1;
    // Update apriori error covariance estimate
    pt_t1        = F*pt1_t1*F + rho2*Q;

    // ========
    // Update 
    // ========

    // Innovation
    e              = z - H*xt_t1;

    // Covariance innov
    S              = H*pt_t1*H + R;
    Sinv           = 1/S;

    // Update Kalman gain
    kt              = pt_t1*H * Sinv;

    // Update aposteriori state estimate
    xt_t            = xt_t1 + kt * e;

    // Update aposteriori error covariance estimate

    //pt_t            = ( diag(1,length(xt_t)) - kt%*%H) %*% pt_t1;
    pt_t            = pt_t1 - kt * H * pt_t1;

    // ========
    // Likehood + CUSUM 
    // ========
    st = -log(det(S)) - 1/(sigma2b)*e*Sinv*e + 1/(sigma2b)*z*z;

		Qt(it) = Qt(it-1) + st;

		if (Qt(it) < 0){
			Qt(it) = 0;
		}
		if (Qt(it) < Qt(it-1)){
			Qt(it) = 0;
		}
		
      
    xt1_t1 = xt_t;
    pt1_t1 = pt_t;

    }

    return Rcpp::List::create(
    Rcpp::Named("Qt") = Qt
    ) ;

}
