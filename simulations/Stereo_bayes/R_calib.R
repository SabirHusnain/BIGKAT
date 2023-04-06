library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

getRT = function(r1, r2, r3, t1, t2, t3){
  
    R_x = matrix(c(1,0,0, 0, cos(r1), -sin(r1), 0, sin(r1), cos(r1)), nrow = 3, ncol = 3, byrow = TRUE)
    
    R_y = matrix(c(cos(r2), 0, sin(r2), 0, 1, 0, -sin(r2), 0, cos(r2)), nrow = 3, ncol = 3, byrow = TRUE)
    
    R_z = matrix(c(cos(r3), -sin(r3), 0, sin(r3), cos(r3), 0, 0, 0, 1), nrow = 3, ncol = 3, byrow = TRUE)
    
    R = R_x %*% R_y %*% R_z
    
    RT = cbind(R, c(t1, t2, t3)) #Add the translation vector
    
    return(RT[,c(1:2, 4)])
}

chessboard = as.matrix(read.csv('chessboard.csv'))
M = matrix(c(896,0,640,0,504,360,0,0,1), nrow = 3, ncol = 3, byrow = TRUE)

rotations = list(c(0.25, -0.1, -2.0), c(-.3, 3, -0.2), c(0.1, -.25, 0))
translations = list(c(25, 50, 600), c(-80, 60, 700), c(98, -25, 400))

proj_all = array(NA, dim = c(length(rotations), nrow(chessboard), 3))

for (i in 1:length(rotations)){
 
  RT = getRT(rotations[[i]][1], rotations[[i]][2], rotations[[i]][3], translations[[i]][1],translations[[i]][2] ,translations[[i]][3])
  
  H = M %*% RT
  
  proj = matrix(NA, nrow=nrow(chessboard), ncol=3)
  
  for (point in 1:nrow(chessboard)){
    
    proj[point,] = H %*% chessboard[point,]
    #proj[point,] = proj[point,] 
    
  } 
  proj = proj /  proj[,3]
  proj_all[i,,] = proj
  #proj_all[[i]] = proj
 
  
}

stan_data = list (images = as.array(proj_all), J = nrow(chessboard), P = dim(proj_all)[1], chessboard = chessboard)

fit <- stan(
  file = "camera_calib_Homog_M.stan",  # Stan program
  data = stan_data,    # named list of data
  chains = 1,             # number of Markov chains
  iter = 50,            # total number of iterations per chain
  cores = 1,              # number of cores (using 1 just for the vignette)
  refresh = 10          # show progress every 'refresh' iterations
)