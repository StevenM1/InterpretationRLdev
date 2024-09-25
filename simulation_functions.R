### functions for RL simulations
softmax <- function(x, beta) {
  exp(beta*x)/(sum(exp(beta*x)))
}

# simulation functions
simulate_RL_softmax <- function(alpha, beta, nTrials, pReward, rewardSensitivity=1, simulateChoice=TRUE) {
  ## assuming a 2-alternative forced choice, c.f. the figure of the paradigm in the draft
  ## so two choice options (and Q-values), each associated with a reward probability
  # latent vars
  Q <- matrix(0, nrow=nTrials, ncol=2)                # qvalues for choice1, choice2
  choices <- PE <- matrix(NA, nrow=nTrials, ncol=1)   # Keep track of rewards, choices, PEs per trial
  rewards <- pChoices <- matrix(NA, nrow=nTrials, ncol=2)
  
  for(i in 1:nTrials) {
    # simulate choice
    if(simulateChoice) {
      pChoice <- softmax(Q[i,], beta)
    } else {
      pChoice <- c(1,0)
    }
    pChoices[i,] <- pChoice
    choice <- rbinom(1, 1, pChoice[2])+1
    notchoice <- 3-choice
    
    # simulate reward
    reward <- rbinom(1, 1, pReward[choice])
    reward <- ifelse(reward==0, -1, 1)
    
    choices[i] <- choice
    rewards[i,choice] <- reward
    #print(paste0('Q1: ', Q[i,1], ', Q2: ', Q[i,2], ', Choice: ', choice, ', reward: ', reward))
    
    # prediction error
    PE[i] <- rewardSensitivity*reward-Q[i,choice]
    
    # update Q-value
    if(i < nTrials) {
      Q[i+1,choice] <- Q[i, choice] + alpha*PE[i]
      Q[i+1, notchoice] <- Q[i, notchoice]
    }
  }
  
  return(list(Q=Q, PE=PE, choices=choices, rewards=rewards, pChoices=pChoices))
}

plotRL <- function(Q, PE, choices) {
  # plot for sanity check
  plot(Q[,1], col=2, type='l', ylim=range(Q))
  lines(Q[,2], col=3)
  plot(PE, col=choices+1)
}

# fitting functions / likelihood
ll_func <- function(pars, pnames, rewards, choices, min.like=1e-20, constants=NULL) {
  names(pars) <- pnames
  if(!is.null(constants)) {
    pars <- c(pars, constants)
  }
  alpha <- pars[['alpha']]
  beta <- pars[['beta']]
  gamma <- pars[['gamma']]
  
  ## emcAdapt is a self-written library that implements an RL update loop in C -- allows for quick updating
  updated <- emcAdapt::adapt.c.emc(rewards*gamma, arguments=list(startValues=c(0,0), learningRates=matrix(alpha, nrow=nrow(rewards), ncol=2)))
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


#### Neural data functions
## some functions for simulating/fitting MRI data
make_design_matrix <- function(nTrials, PE, TR=1, duration=.001, oversampling=50) {
  frame_times <- seq(0, nTrials*4, TR)  # assume 4 seconds per trial, for no good reason
  events <- data.frame('onset'=seq(0, nTrials*4, length.out=nTrials), 'trial_type'='PE', 'duration'=duration, 'modulation'=PE)
  X <- as.matrix(make_fmri_design_matrix(frame_times=frame_times, events=events, oversampling=oversampling))  # design matrix
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

