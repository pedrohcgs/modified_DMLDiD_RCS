# modifiedDMLDiD
Modifying DMLDiD :Double/debiased machine learning for difference-in-differences models.

# previous work
## main paper
[1] Chang, Neng-Chieh. (2020). [Double/debiased machine learning for difference-in-differences models](https://academic.oup.com/ectj/article/23/2/177/5722119#247745047). The Econometrics Journal 23.2 : 177–191
- This paper proposes the following two models:
  - repeated outcome (dmldid_ro)：
    - I believe this to be a theoretically valid and excellent achievement.
    - However, there are several deficiencies in the implementation of Chang (2020). In addition, the simulation data in the paper were not suitable to demonstrate this theory. These points have been corrected and presented in [My blog](https://medium.com/@masa_asami/double-debiased-ml-for-did-1-fd08bebcf033).
  - repeated cross-section  (dmldid_rcs):
    - I do not believe this model is adequate. This repository will examine and attempt to modify this.

## sub papers
[2] Abadie A. (2005). Semiparametric difference-in-differences estimators, Review of Economic Studies, 72, 1–19.
- References in Chang (2020)
- Proposes an IPW estimator for DiD.

[3] Chernozhukov V., D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, J. Robins (2018). Double/debiased machine learning for treatment and structural parameters, Econometrics Journal, 21, C1–C68.
- original DML

# What is wrong with dmldid_rcs ?
## dmldid_rcs's ATT in Chang (2020)

- Chang (2020) defines the ATT estimator for repeated cross-section data as follows:
- Y : outcome
- T : binary value (post =1 / pre = 0)
- D : binary value (treated group =１/ control group = 0)
- X : Covariates
- p̂k : D's average（cross-fitting）
- λ̂k : T's average（cross-fitting）
- ĝk(X) : propensity socre（cross-fitting）


![
\tilde{\theta }_{k}=\frac{1}{n}\sum _{i\in I_{k}}\frac{D_{i}-\hat{g}_{k}\left(X_{i}\right)}{\hat{p}_{k}\hat{\lambda }_{k}\left(1-\hat{\lambda }_{k}\right)\left(1-\hat{g}_{k}\left(X_{i}\right)\right)}\times \left(\left(T_{i}-\hat{\lambda }_{k}\right)Y_{i}-\hat{\ell }_{2k}\left(X_{i}\right)\right)](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bwhite%7D%5Clarge+%5Cdisplaystyle+%0A%5Ctilde%7B%5Ctheta+%7D_%7Bk%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum+_%7Bi%5Cin+I_%7Bk%7D%7D%5Cfrac%7BD_%7Bi%7D-%5Chat%7Bg%7D_%7Bk%7D%5Cleft%28X_%7Bi%7D%5Cright%29%7D%7B%5Chat%7Bp%7D_%7Bk%7D%5Chat%7B%5Clambda+%7D_%7Bk%7D%5Cleft%281-%5Chat%7B%5Clambda+%7D_%7Bk%7D%5Cright%29%5Cleft%281-%5Chat%7Bg%7D_%7Bk%7D%5Cleft%28X_%7Bi%7D%5Cright%29%5Cright%29%7D%5Ctimes+%5Cleft%28%5Cleft%28T_%7Bi%7D-%5Chat%7B%5Clambda+%7D_%7Bk%7D%5Cright%29Y_%7Bi%7D-%5Chat%7B%5Cell+%7D_%7B2k%7D%5Cleft%28X_%7Bi%7D%5Cright%29%5Cright%29)


![\hat{\ell }_{2k}\left(x_{i}\right)\equiv q_{i}^{\prime }\hat{\beta }_{2k}
](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bwhite%7D%5Clarge+%5Ctextstyle+%5Chat%7B%5Cell+%7D_%7B2k%7D%5Cleft%28x_%7Bi%7D%5Cright%29%5Cequiv+q_%7Bi%7D%5E%7B%5Cprime+%7D%5Chat%7B%5Cbeta+%7D_%7B2k%7D%0A)


![\hat{\beta }_{2k}\in \arg \min _{\beta \in \mathbb {R}^{p}}\left[\frac{1}{M_{k}}\sum _{i\in I_{k}^{c}}\left(1-D_{i}\right)\left(\left(T_{i}-\hat{\lambda }_{k}\right)Y_{i}-q_{i}^{\prime }\beta \right)^{2}\right]+\frac{\lambda _{2k}}{M_{k}}\parallel \hat{\Upsilon }_{2k}\beta \parallel _{1}
](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bwhite%7D%5Clarge+%5Cdisplaystyle+%5Chat%7B%5Cbeta+%7D_%7B2k%7D%5Cin+%5Carg+%5Cmin+_%7B%5Cbeta+%5Cin+%5Cmathbb+%7BR%7D%5E%7Bp%7D%7D%5Cleft%5B%5Cfrac%7B1%7D%7BM_%7Bk%7D%7D%5Csum+_%7Bi%5Cin+I_%7Bk%7D%5E%7Bc%7D%7D%5Cleft%281-D_%7Bi%7D%5Cright%29%5Cleft%28%5Cleft%28T_%7Bi%7D-%5Chat%7B%5Clambda+%7D_%7Bk%7D%5Cright%29Y_%7Bi%7D-q_%7Bi%7D%5E%7B%5Cprime+%7D%5Cbeta+%5Cright%29%5E%7B2%7D%5Cright%5D%2B%5Cfrac%7B%5Clambda+_%7B2k%7D%7D%7BM_%7Bk%7D%7D%5Cparallel+%5Chat%7B%5CUpsilon+%7D_%7B2k%7D%5Cbeta+%5Cparallel+_%7B1%7D%0A)


- l2k is the following ML model (Chang(2020) uses Lasso, but essentially any ML is OK)
  - Supervised label：(Ti−λ̂k)Yi
  - features：covariates X（X with arbitrary transformations as q）
  - training data：untreated data only 
  - with cross-fitting
  
![image](https://user-images.githubusercontent.com/16971400/198495736-bc34e71b-a1bd-443f-ab83-2347c9ab0f28.png)


## l2k is an unrealistic prediction task
- The label is positive if T=1 and negative if T=0. The sign is easily reversed by a change in T alone.
- On the other hand, X(q) does **not** necessarily contain time-dependent variables.　And even if it does contain such variables, it is difficult to properly predict the above label with a linear model (Lasso) such as Chang (2020)
  - For example, if the covariates are not time-dependent (e.g., demographic information such as male/female, race, etc.), prediction of this label is not possible (although it can be learned = no error occurs, but the prediction is meaningless)
  - Even if the covariates include time-dependent variables, a large number of time-dependent *time-independent interaction terms must be thrown in. Still, it is almost impossible to predict the label such that it flips 180 degrees positive or negative due to unobserved variables (for l2k, T is unobserved)
- The (Ti-λ̂k) part need not be included in the prediction task. It should only be designed to estimate the latent outcome of Y
![image](https://user-images.githubusercontent.com/16971400/198539758-ae7d2ccd-3e36-42e6-bf6d-5db2d0079dcd.png)

# My solution

- First, divide l2k into two parts l2k_t1, l2k_t0 are arbitrary machine learning models
![\hat{\ell }_{2k}^{t } = E[Y_i   |  T=t,D=0, X_i]](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bwhite%7D%5Clarge+%5Cdisplaystyle+%5Chat%7B%5Cell+%7D_%7B2k%7D%5E%7Bt+%7D+%3D+E%5BY_i+++%7C++T%3Dt%2CD%3D0%2C+X_i%5D)
  - l2k_t1 = E[Y | T=1, D=0, X] 
  - l2k_t0 = E[Y | T=0, D=0, X]
- With this l2k_t1, l2k_t0, the ATT can be modified as follows.
  - Doubly robust in propensity score and l2k (outcome model) is achieved.
  - cross-fitting

![\tilde{\theta }_{k}=\frac{1}{n}\sum _{i\in I_{k}}\frac{D_{i}-\hat{g}_{k}\left(X_{i}\right)}{\hat{p}_{k}\hat{\lambda }_{k}\left(1-\hat{\lambda }_{k}\right)\left(1-\hat{g}_{k}\left(X_{i}\right)\right)}
\times
\left(T_{i}-\hat{\lambda }_{k}\right) 
 \left(Y_{i}-\hat{\ell }_{2k}^{t}\left(X_{i}\right)\right)
](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bwhite%7D%5Clarge+%5Cdisplaystyle+%5Ctilde%7B%5Ctheta+%7D_%7Bk%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum+_%7Bi%5Cin+I_%7Bk%7D%7D%5Cfrac%7BD_%7Bi%7D-%5Chat%7Bg%7D_%7Bk%7D%5Cleft%28X_%7Bi%7D%5Cright%29%7D%7B%5Chat%7Bp%7D_%7Bk%7D%5Chat%7B%5Clambda+%7D_%7Bk%7D%5Cleft%281-%5Chat%7B%5Clambda+%7D_%7Bk%7D%5Cright%29%5Cleft%281-%5Chat%7Bg%7D_%7Bk%7D%5Cleft%28X_%7Bi%7D%5Cright%29%5Cright%29%7D%0A%5Ctimes%0A%5Cleft%28T_%7Bi%7D-%5Chat%7B%5Clambda+%7D_%7Bk%7D%5Cright%29+%0A+%5Cleft%28Y_%7Bi%7D-%5Chat%7B%5Cell+%7D_%7B2k%7D%5E%7Bt%7D%5Cleft%28X_%7Bi%7D%5Cright%29%5Cright%29%0A)

# Simulation result
## simulation data
The following simulation data were created:
- repeated cross-section data
- true ATT := 3
- dim(X) := 10
- N = 500
## result: modified-DMLDiD is better than Chang(2020)
- The notebook is [here](https://github.com/MasaAsami/modified_DMLDiD_RCS/blob/main/notebooks/DMLDiD_RCS_with_SIMDATA.ipynb).
- ture ATT = 3
![image](https://user-images.githubusercontent.com/16971400/198532873-c2a574f5-1625-4825-987c-3fdb9b5132ba.png)



