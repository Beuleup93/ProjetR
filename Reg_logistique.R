# Library for showing plot
library(ggplot2)

#_______________________________________________________________________________
# GENERATION DONNÉES LOGISTIQUE
set.seed(100)
n <- 200
p <- 6

theta = c(1,6,3.5,rep(0,p-2)) # or theta = runif(7)
X <- cbind(1,matrix(rnorm(n*p),n,p)) #  6 Variables quantitative
Z <- X %*% theta # combinaison lineare de variable
fprob <- ifelse(Z<0, exp(Z)/(1+exp(Z)),1/(1+exp(-Z))) # Calcul des probas d'affectation
y<- rbinom(n,1,fprob) # La variable réponse est une variable aléatoire de bernouli

class(X) # matrix
class(y) # vecteur d'entier
class(theta) # vecteur de numeric

dim(X) # X est une matrice
length(y) # vecteur
length(theta)

# Visualisation des données
# combine the data

data <- cbind(X,y)
colnames(data) <- c("biais","x1","x2","x3","x4","x5","x6", "y")
data = as.data.frame(data)
# visualize the data:
ggplot(data) + 
  geom_point(aes(x=x1, y=x2, color = as.character(y)), size = 2) + 
  scale_colour_discrete(name  ="Label") + 
  ylim(-3, 3) + coord_fixed(ratio = 1) +
  ggtitle('Data to be classified') +
  theme_bw(base_size = 12) +
  theme(legend.position=c(0.85, 0.87))

#_______________________________________________________________________________

#_______________________________________________________________________________
# CREATION DES FONCTIONS POUR LA REGRESSION
sigmoid <- function(z){
  return (1/(1+exp(-z)))
}

#Loss Logistic (returne le cout)
log_loss <- function(theta, X, y){
  # theta: vecteur contenant nos paramettres à estimer taille(p+1,1) avec p le nombre de variables explicatives
  # X : Matrice des variables explicative + la colonne de bias taille(n, p+1) 
  # y: Varaible cible sous forme matrice de taille (n*1)
  #_______________________
  # m =nombre de ligne d'apprentissage
  n <- length(y) 
  # PI: notre probabilité d'affectation à la classe positive (produit matriciel (n, p+1) (p+1,1) ==> (n,1))
  PI <- sigmoid(X %*% theta) 
  # Calcul de la fonction de cout ((n,1) (n,1) => (1,1)scalaire)
  J <- (t(-y)%*%log(PI)-t(1-y)%*%log(1-PI))/n
  return(J)
}

#gradient function (retourne le vecteur gradient)
gradient <- function(theta, X, y){
  n <- length(y) 
  PI <- sigmoid(X%*%theta)
  gradient <- (t(X)%*%(PI - y))/n
  return (as.vector(gradient))
}

#######________DESCENTE DE GRADIENT STOCKASTIQUE(BATCH)

# Algorithme de la descente de gradient
gradient_descent<- function(X,y,theta, leaning_rate=0.1, max_iter=100, tolerance=1e-04){
  # Controle du taux d'apprentissage
  if (leaning_rate <= 0){
    stop("'learn_rate' doit etre superieur à zero")
  }
  # Controle de la tolerance
  if (tolerance <= 0){
    stop("'tolerance' doit etre superieur à zero")
  }
  # Controle du max iteratons
  if (max_iter <= 0){
    stop("'max_iter' doit etre superieur à zero")
  }
  # Controle de dimension
  if (dim(X)[1] != length(y)){
    stop("les dimensions de 'x' et 'y' ne correspondent pas")
  }
  #remove NA rows
  X <- na.omit(X)
  y <- na.omit(y)
  # Vecteur de cout
  cost_vector = c()
  m = nrow(X)
  # controle de convergence
  converge = FALSE
  iter <- 0
  while((iter < max_iter) && (converge == FALSE) ){
    #iteration suivante
    iter <- iter +1
    # Calcul du vecteur probas
    PI <- sigmoid(X%*%theta) 
    # Calcul du cout
    cost = log_loss(theta, X, y)
    # Historisation de la fonction de cout
    cost_vector = c(cost_vector, cost)
    # Mise à jour du theta
    gradient = gradient(theta, X, y)
    old_theta = theta
    theta = theta - leaning_rate*gradient
    # Controle de convergence
    if (sum(abs(theta-old_theta)) < tolerance){
      converge <- TRUE
    }
  }
  return(list(theta_final = theta, history_cost = cost_vector, nbIter=iter))
}

#######________DESCENTE DE GRADIENT STOCKASTIQUE(MINI BATCH)

# Algorithme de la descente de gradient
gradient_descent_mini_batch<- function(X,y,theta, batch_size=1, random_state=NA_integer_, leaning_rate=0.1, max_iter=100, tolerance=1e-04){
  # Controle du taux d'apprentissage
  if (leaning_rate <= 0){
    stop("'learn_rate' doit etre superieur à zero")
  }
  # Controle de la tolerance
  if (tolerance <= 0){
    stop("'tolerance' doit etre superieur à zero")
  }
  # Controle du max iteratons
  if (max_iter <= 0){
    stop("'max_iter' doit etre superieur à zero")
  }
  # Controle de dimension
  if (dim(X)[1] != length(y)){
    stop("les dimensions de 'x' et 'y' ne correspondent pas")
  }
  # Initialiser le generateur de nombre aleatoire pour rendre reproductible les calculs
  if(!is.na(random_state)){
    set.seed(random_state) # Ceci permet de reproduire le shufle à chaque fois
  }
  # Setting up and checking the size of minibatches
  if( (batch_size <=  0) || (batch_size > dim(X)[1]-1)){
    stop("'Batch size' doit etre compris entre zero et le nombre total d'observations moins 1")
  }
  xy = cbind(X,y)
  #colnames(xy) <- c("biais","x1","x2","x3","x4","x5","x6","y")
  #remove NA rows
  X <- na.omit(X)
  y <- na.omit(y)
  # Vecteur de cout
  cost_vector = c()
  m = nrow(X)
  # controle de convergence
  converge = FALSE
  iter <- 0
  nb_iter_ <- 0
  while((iter < max_iter) && (converge == FALSE) ){
    #iteration suivante
    iter <- iter + 1
    # SHUFLE the dataset
    rows <- sample(nrow(xy))  # Melanger les indices du dataframe xy
    xy <- xy[rows, ] # Utiliser ces indices pour reorganiser le dataframe
    # MINI BATCH 
    for (start in seq(from=1, to=dim(X)[1], batch_size)){
      stop = start + batch_size
      if(stop > dim(X)[1]){
        break
      }
      xBatch = xy[start:stop,-ncol(xy)]
      yBatch = xy[start:stop, ncol(xy)]
      #print(dim(xBatch))
      # Calcul du vecteur probas
      PI <- sigmoid(xBatch%*%theta) 
      # Calcul du cout
      cost = log_loss(theta, xBatch, yBatch)
      # Historisation de la fonction de cout
      cost_vector = c(cost_vector, cost)
      # conteur permettant de suivre le nomre d'element de history
      nb_iter_ = nb_iter_ +1 
      # Mise à jour du theta
      gradient = gradient(theta, xBatch, yBatch)
      new_theta = theta - leaning_rate*gradient
      # Controle de convergence
      #print(sum(abs(new_theta - theta)))
      if (sum(abs(new_theta - theta)) < tolerance){
        converge <- TRUE
        print(sum(abs(new_theta - theta)))
        print(converge)
        break
      }
      theta = new_theta
    }
  }
  return(list(theta_final = theta, history_cost = cost_vector, nbIter=iter, nb_iter_interne = nb_iter_))
}

# y prediction
predict_log <- function(PI){
  return(round(PI, 0))
}

# coef determination
coef_determination <- function(y, ypred){
  
  u = sum((y-ypred)**2)
  v = sum((y-mean(y))**2)
  return(1-u/v)
}


#_______________________________________________________________________________
# Teste des fonctions 
X = as.matrix(X)
y = as.vector(y)
log_loss(theta = theta, X=X, y=y)
# Teste de gradient
gradient(theta = theta, X=X, y=y)
# Comparaison avec le gradient de R
library(numDeriv)
grad(log_loss,theta, y=y, X=X )

# Teste de l'algorithme de gradient

# batch
print(system.time(out <- gradient_descent(X=X, y=y, theta=theta,leaning_rate = 0.7, max_iter = 1000, tolerance = 1e-04)))
xi=1:out$nbIter
yi=out$history_cost
plot(xi, yi, type="l")
# minibatch
print(system.time(out1 <- gradient_descent_mini_batch(X,y,theta, batch_size=10, random_state=1, leaning_rate=0.1, max_iter=1000, tolerance=1e-04)))
# online
print(system.time(out2 <- gradient_descent_mini_batch(X,y,theta, batch_size=1, random_state=1, leaning_rate=0.05, max_iter=100, tolerance=1e-04)))
xi=1:out2$nb_iter_interne
yi=out2$history_cost
plot(xi, yi, type="l")

length(out2$history_cost)
#_______________________________________
xi=1:length(out$history_cost)
yi=out$history_cost
plot(xi, yi, type="l")




# Affichage des parametres de la fonction
#print(system.time(coef_grad <- gradient_simple(x,y,eta=1.0)))
out$nbIter
out1$nb_iter_interne
out2$nbIter
out2$nb_iter_interne
#out$history_cost

# COMPARAISONS DESCENT GRADIENT VS NEWTON RAPHTON (BFGS)
# Gadient descent
grad.coef_batch = out$theta_final
grad.coef_minibatch = out1$theta_final
grad.coef_online= out2$theta_final
# Newton
newton.coef <- optim(theta, log_loss, y=y, X=X, method = "BFGS")$par
# Comparaison
cbind(GradDescBatch=grad.coef_batch, GradDescMiniBatch=grad.coef_minibatch, GradDescOnline=grad.coef_online, BFGS=newton.coef)

# PROBABILITY PREDICTION
#_____________________________ BATCH
theta_final = out$theta_final
Z = X %*% theta_final
prob_pred <- 1/(1+exp(-Z))
y_pred <- ifelse(prob_pred>0.5,1,0)
coef_determination(y=y,ypred = y_pred)

# ____________________________ MINI BATCH
theta_final1 = out1$theta_final
Z1 = X %*% theta_final1
prob_pred1 <- 1/(1+exp(-Z))
y_pred1 <- ifelse(prob_pred1>0.5,1,0)
coef_determination(y=y,ypred = y_pred1)

#__________________________ ONLINE
theta_final2 = out2$theta_final
Z2 = X %*% theta_final2
prob_pred2 <- 1/(1+exp(-Z))
y_pred2 <- ifelse(prob_pred2>0.5,1,0)
coef_determination(y=y,ypred = y_pred2)

# BINARY PREDICTION
mat = cbind(TrueY=y, y_pred_batch = y_pred, y_pred_minibatch = y_pred1,y_pred_online = y_pred2, pred_prob=prob_pred,pred_prob=prob_pred1,pred_prob=prob_pred2)
colnames(mat) <- c('y','y_pred_batch','y_pred_minibatch','y_pred_online','probas_batch','probas_mini','probas_online')


# Taux de reconnaissance
coef_determination(y=y,ypred = y_pred)

#_______________________________________________________________________________
#decision boundary visualization
# generate a grid for decision boundary, this is the test set
grid <- expand.grid(seq(0, 3, length.out = 100), seq(0, 3, length.out = 100))
y_prd <- predict_log(prob_pred)
gridPred = cbind(grid, y)

ggplot() +   
  geom_point(data = data, aes(x=x1, y=x2, color = as.character(y)), size = 2, show.legend = F) + 
  geom_tile(data = gridPred, aes(x = grid[, 1],y = grid[, 2], fill=as.character(y_prd)), alpha = 0.3, show.legend = F)+ 
  ggtitle('Decision Boundary for Logistic Regression') +
  coord_fixed(ratio = 1) +
  theme_bw(base_size = 12) 








