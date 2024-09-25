library(emcAdapt)  ## <- package for fast RL updating, using some C-code
library(DEoptim)
library(pracma)
source('fMRI.R') ## <- some code to generate fMRI design matrices and the like

#####################
#####################
##### Functions #####
#####################
#####################

# Functions to create design matrix, neural signal and to fit fMRI GLM
make_design_matrix <- function(nTrials, PE, TR=1) {
  frame_times <- seq(0, nTrials*4, TR)  # assume 4 seconds per trial, for no good reason
  events <- data.frame('onset'=seq(0, nTrials*4, length.out=nTrials), 'trial_type'='PE', 'duration'=0.001, 'modulation'=PE)
  X <- as.matrix(make_fmri_design_matrix(frame_times=frame_times, events=events))  # design matrix
  X
}
make_fMRI_signal <- function(X, beta_PE=1, beta_intercept=0, sd_noise=.1) {
  betas <- c(beta_PE, beta_intercept)
  signal <- X%*%betas + rnorm(nrow(X), 0, sd_noise)   # some noise, maybe not needed?
  signal
}
fit_GLM <- function(X, signal) {
  pinv(t(X) %*% X)%*%t(X)%*%signal  ## analytic GLM solution
}

# Functions to simulate behavioral data
softmax <- function(x, beta) {
  exp(beta*x)/(sum(exp(beta*x)))
}
simulate_RL_softmax_dynamic <- function(alpha, beta, nTrials, pReward, rewardSensitivity=1, decay) {
  ## assuming a 2-alternative forced choice, c.f. the figure of the paradigm in the draft
  ## so two choice options (and Q-values), each associated with a reward probability
  # latent vars
  Q <- matrix(0, nrow=nTrials, ncol=2)                # qvalues for choice1, choice2
  choices <- PE <- matrix(NA, nrow=nTrials, ncol=1)   # Keep track of rewards, choices, PEs per trial
  rewards <- matrix(NA, nrow=nTrials, ncol=2)
  alphaAcc <- numeric(nTrials)
  
  for(i in 1:nTrials) {
    # simulate choice
    pChoice <- softmax(Q[i,], beta)
    choice <- rbinom(1, 1, pChoice[2])+1
    notchoice <- 3-choice
    
    # simulate reward
    reward <- rbinom(1, 1, pReward[choice])
    
    choices[i] <- choice
    rewards[i,choice] <- reward
    
    # prediction error
    PE[i] <- rewardSensitivity*reward-Q[i,choice]
    
    # determine learning rate
    alphaAcc[i] <- alpha * i^(-decay)
    
    # update Q-value
    if(i < nTrials) {
      Q[i+1,choice] <- Q[i, choice] + alphaAcc[i]*PE[i]
      Q[i+1, notchoice] <- Q[i, notchoice]
    }
  }
  
  return(list(Q=Q, PE=PE, choices=choices, rewards=rewards, alpha=alphaAcc))
}
simulate_RL_softmax_static <- function(alpha, nTrials, rewardSensitivity=1, choices, rewards) {
  ## assuming a 2-alternative forced choice, c.f. the figure of the paradigm in the draft
  ## so two choice options (and Q-values), each associated with a reward probability
  # latent vars
  Q <- matrix(0, nrow=nTrials, ncol=2)                # qvalues for choice1, choice2
  PE <- matrix(NA, nrow=nTrials, ncol=1)   # Keep track of rewards, choices, PEs per trial
  
  for(i in 1:nTrials) {
    choice <- choices[i]
    notchoice <- 3-choice
    
    # prediction error
    PE[i] <- rewardSensitivity*rewards[i,choice]-Q[i,choice]
    
    # update Q-value
    if(i < nTrials) {
      Q[i+1,choice] <- Q[i, choice] + alpha*PE[i]
      Q[i+1, notchoice] <- Q[i, notchoice]
    }
  }
  
  return(list(Q=Q, PE=PE))
}

## likelihood function of an RL model
ll_func <- function(pars, pnames, rewards, choices, min.like=1e-20, constants=NULL) {
  names(pars) <- pnames
  if(!is.null(constants)) {
    pars <- c(pars, constants)
  }
  alpha <- pars[['alpha']]
  beta <- pars[['beta']]
  gamma <- pars[['gamma']]
  
  ## emcAdapt is a self-written library that implements an RL update loop in C -- allows for quick updating
  updated <- emcAdapt::adapt.c.emc(rewards*gamma, arguments=list(startValues=c(0,0), learningRates=matrix(alpha, nrow=nTrials, ncol=2)))
  Q <- updated$adaptedValues
  PEs <- updated$predictionErrors
  
  # softmax
  PP <- exp(Q*beta)
  PP <- PP/apply(PP, 1, sum)
  
  LL <- rep(min.like, nrow(rewards))
  LL[as.numeric(choices)==1] <- PP[as.numeric(choices)==1, 1]
  LL[as.numeric(choices)==2] <- PP[as.numeric(choices)==2, 2]
  
  if(any(is.na(LL))) return(Inf)
  -sum(log(LL))
}

#################################
##### Simulate dynamic data #####
#################################

# Simulate behavioral data with dynamic (decaying) learning rate
nTrials = 1e3
LR0 = 0.8                         # initial LR
decay = .1                        # LR decay (the higher the faster it decays)
alpha = LR0*(1:nTrials)^(-decay)  # resulting LR across trials
beta = 1                          # inverse temperature
pReward <- c(.8, .2)              # reward probabilities for choice1, choice2; assuming a single stimulus

set.seed(338444)
# Simulate dynamic data
out.true.dynamic = simulate_RL_softmax_dynamic(LR0, beta, nTrials, pReward, decay=decay)
# Get output
Q.true.dynamic <- out.true.dynamic$Q
PE.true.dynamic <- out.true.dynamic$PE
choices.true.dynamic <- out.true.dynamic$choices
rewards.true.dynamic <- out.true.dynamic$rewards

# what's the LL of the true parameters?
ll.true <- ll_func(c(alpha, beta, 1), pnames=c('alpha', 'beta', 'gamma'), rewards=rewards.true.dynamic, choices=choices.true.dynamic, constants=NULL)

############################
##### Fit static model #####
##### on dynamic data ######
############################

# fit -- we need to fix one parameter to a constant here
pnames=c('alpha', 'beta', 'gamma')
constants=c('gamma'=1)
pnames <- pnames[!grepl(names(constants), pnames)]
out.fit.incorrect <- DEoptim(ll_func, 
                             lower=setNames(c(0,0), pnames), upper=setNames(c(1,10), pnames), 
                             rewards=rewards.true.dynamic, choices=choices.true.dynamic, pnames=pnames, constants=constants)

c('LR0.true'=LR0,'decay.true'=decay,'meanalpha.true'=mean(alpha),'ll.true'=ll.true, setNames(out.fit.incorrect$optim$bestmem, paste0(names(out.fit.incorrect$optim$bestmem), '.fit')), 'll.fit'=out.fit.incorrect$optim$bestval)

################################
##### Simulate static data #####
###### with recovered LR #######
##### from previous step #######
################################

# Simulate static data
out.true.static = simulate_RL_softmax_dynamic(out.fit.incorrect$optim$bestmem["alpha"], beta, nTrials, pReward, decay=0)
# Get output
Q.true.static <- out.true.static$Q
PE.true.static <- out.true.static$PE
choices.true.static <- out.true.static$choices
rewards.true.static <- out.true.static$rewards

# what's the LL of the true parameters?
ll.true <- ll_func(c(alpha, beta, 1), pnames=c('alpha', 'beta', 'gamma'), rewards=rewards.true.static, choices=choices.true.static, constants=NULL)

############################
##### Fit static model #####
###### on static data ######
############################

# fit -- we need to fix one parameter to a constant here
pnames=c('alpha', 'beta', 'gamma')
constants=c('gamma'=1)
pnames <- pnames[!grepl(names(constants), pnames)]
out.fit.correct <- DEoptim(ll_func, 
                           lower=setNames(c(0,0), pnames), upper=setNames(c(1,10), pnames), 
                           rewards=rewards.true.static, choices=choices.true.static, pnames=pnames, constants=constants)

c('alpha.true'=out.fit.incorrect$optim$bestmem["alpha"],'ll.true'=ll.true, setNames(out.fit.correct$optim$bestmem, paste0(names(out.fit.correct$optim$bestmem), '.fit')), 'll.fit'=out.fit.correct$optim$bestval)

#######################################
##### Simulate true neural data #######
#######################################

# Simulate true neural data
PE.dynamic <- out.true.dynamic$PE
PE.static <- out.true.static$PE

X.dynamic <- make_design_matrix(nTrials, PE.dynamic)
X.static <- make_design_matrix(nTrials, PE.static)

signal.dynamic <- make_fMRI_signal(X.dynamic, beta_PE=1)
signal.static <- make_fMRI_signal(X.static, beta_PE=1)

##########################
##### Fit fMRI GLM #######
##########################

# Make design matrix from computational output for neural model fitting
PE.fit.correct <- simulate_RL_softmax_static(out.fit.correct$optim$bestmem["alpha"],nTrials,choices=out.true.static$choices, rewards=out.true.static$rewards)$PE
PE.fit.incorrect <- simulate_RL_softmax_static(out.fit.incorrect$optim$bestmem["alpha"],nTrials,choices=out.true.dynamic$choices, rewards=out.true.dynamic$rewards)$PE

X.fit.correct.static <- make_design_matrix(nTrials, PE.fit.correct)
X.fit.incorrect.static <- make_design_matrix(nTrials, PE.fit.incorrect)

# Fit neural data with PEs from correctly specified model
betas_correct_dynamic = fit_GLM(X.dynamic, signal.dynamic)
betas_correct_static = fit_GLM(X.fit.correct.static, signal.static)
betas_misspecified = fit_GLM(X.fit.incorrect.static, signal.dynamic)

#############################
##### Plot scenario 7 #######
#############################

# Plotting settings
volumesPerTrial <- 4
nTrialsToPlot <- 13
nVolumes = nTrialsToPlot*volumesPerTrial
startVolume = 104    # plot somewhere after the initial learning phase
startTrial = startVolume/volumesPerTrial
ylims <- c(-3, 3)

palette(c("black", 'lightslateblue', 'dark orange'))

for(figtype in c('pdf', 'png')) {
  if(figtype == 'pdf') pdf(file='./figures/scenario7.pdf', width=9.5, height=3.5)
  if(figtype == 'png') png(file='./figures/scenario7.png', width=9.5, height=3.5, units='in', res=175)
  
  l <- layout(matrix(c(1,1,2,2,3,4,5,6,1,1,2,2), nrow=3, byrow = TRUE), heights=c(.01, .9, .01))
  plot.new()
  mtext(expression(bold('A) Correct specification')), line=2.5, cex=.66*1.5)
  plot.new()
  mtext(expression(bold('B) Misspecification')), line=2.5, cex=.66*1.5)
  par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
  plot(1:nTrials, out.true.dynamic$alpha, col=2, lwd=2,type="l",ylab=expression(alpha),bty='l',xlab="Trial",ylim=c(0.4,0.8), cex.lab=1.2)
  abline(h=out.fit.incorrect$optim$bestmem["alpha"], col=3, lwd=2)
  legend('topright', c('Participant A','Participant B'), bty='n', lwd=c(2,2), col=c(2,3), lty=c(1,1),cex=1)
  par(mar=c(3,4,3,.5), las=1, mgp=c(2,.5,0))
  barplot(c(betas_correct_dynamic[1], betas_correct_static[1]), ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1), col=c(2,3),cex.lab=1.2, cex.axis=1.2)
  
  barplot(c(out.fit.incorrect$optim$bestmem["alpha"],out.fit.correct$optim$bestmem["alpha"]), ylab=expression(paste('Estimated ', alpha)), xlab='Participant', names.arg=c('A', 'B'), col=c(2,3),ylim=c(0,.7),axes=F, cex.lab=1.2)#, main='Misspecification')
  axis(2,at=seq(0,0.7,by=.1),cex.axis=1.2)
  barplot(c(betas_misspecified[1], betas_correct_static[1]), ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1), col=c(2,3),cex.lab=1.2, cex.axis=1.2)
  dev.off()
}
