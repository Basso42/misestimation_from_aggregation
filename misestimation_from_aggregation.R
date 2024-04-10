#==============================================================================#
########################### replication example ################################
#==============================================================================#


# 1. install R packages
# 2. example network
# 3. calculate overlap coefficients IOC and OOC
# 4. create synthetic shocks
# 5. calculate losses from shock propagation



#==============================================================================#
######################### 1. install R packages   ##############################
#==============================================================================#

# set paths to zip (Windows) or tar.gz (Linux/Mac) files of GLcascade and fastcascade packages
path <- "C:/Users/CD/Documents/GitHub/misestimation_from_aggregation/" # put your own path here 

# install packages from source
install.packages(paste0(path, "GLcascade_0.9.3.1.zip"), 
                 repos = NULL, type = "win.binary")

install.packages(paste0(path, "fastcascade_0.9.3.1.zip"), 
                 repos = NULL, type = "win.binary")

# load the packages
library(GLcascade)
library(fastcascade)
library(Matrix)
library(igraph)
library(colorspace)
library(parallel)

??GLcascade::GL_cascade


#==============================================================================#
########################### 2. minimal example ################################# 
#==============================================================================#

# network W from Fig. 1 

# number of firms
n <- 11

# sector affiliation vector
p <- c(1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5)


# firm-level network edgelist: supplier, buyer, linkweight
el <- rbind(c(3, 6, 1),
            c(4, 6, 1),
            c(4, 7, 1), 
            c(5, 7, 1),
            c(6, 1, 1),
            c(6, 2, 1),
            c(7, 10, 1), 
            c(7, 11, 1),
            c(8, 1, 1),
            c(8, 2, 1),
            c(9, 11, 1))

W <- Matrix::sparseMatrix(i = el[,1], j = el[,2], x = el[,3], dims = c(n,n))

plot(graph_from_adjacency_matrix(as.matrix(W)), edge.arrow.size = 0.3, 
     vertex.color = qualitative_hcl(length(unique(p)), palette = "Dark 3")[as.numeric(p)], 
     main = "W")

### create the sector matrix Z

# sectors contained
sectors <- sort(unique(p))

# number of sectors
m <- length(sectors)

sectors_cons <- 1:length(sectors)
names(sectors_cons) <- sectors
p_cons <- unname(sectors_cons[as.character(p)])

# sector aggregation matrix
psup <- Matrix::sparseMatrix(i = 1:n,
                             j = p_cons,
                             dims = c(n, m))

# aggregate to sectors
Z <- t(psup) %*% W %*% psup

g_Z <- graph_from_adjacency_matrix(as.matrix(Z),  weighted = TRUE)
plot(g_Z, 
     edge.arrow.size = 0.4, edge.width = E(g_Z)$weight,
     vertex.color = qualitative_hcl(m, palette = "Dark 3"), 
     main = "Z")

#==============================================================================#
########################### 3. calculate overlaps ############################## 
#==============================================================================#

# load file containing functions to perform aggregation and similarity calculation
source(paste0(path, "calculate_io_vector_overlaps.R"))


sim_measures <- c("jaccard", 
                  "overlap_relative", 
                  "cosine")

# lower and upper thresold for in- and out-degree buckets in Fig2
threshes_l <- c(1,   6, 16, 36)
threshes_u <- c(5,  15, 35,  10^6)

# number of cores for parallelization 
ncores = 2


# input vector matrix, sectors times firms matrix, element k,i amount i bought from sector k
iv_mat <- aggregate_suppliers(W, p)


# buyer (output) vector matrix, sectors times firms matrix, element k,i amount i sold to sector k
bv_mat <- aggregate_buyers(W, p)


# list that for each sector contains a matrix with pairwise similarities for firms within the sector, firms with empty vectors are dropped
pw_sim_mat_sector_list_inputs <- calc_pw_sim_mat_list(calc_sec_iv_mats(iv_mats = iv_mat, 
                                                                       sector_vec =  p, 
                                                                       input_threshold = c(threshes_l[1], threshes_u[1]), 
                                                                       rm_na_nace = m+1), 
                                                      similarity = "overlap_relative",
                                                      ncores = ncores)


# input vector overlap of firms in sector 3
pw_sim_mat_sector_list_inputs$nace3 # both firms by from sector 2 --> overlap of 1

# input vector overlap of firms in sector 3
pw_sim_mat_sector_list_inputs$nace5 
# firm 10 buys only from sector 3, but firm 11 buys also from sector 4 --> overlap of 0.5


# list that for each sector contains a matrix with pairwise similarities for firms within the sector, firms with empty vectors are dropped
pw_sim_mat_sector_list_outputs <- calc_pw_sim_mat_list(calc_sec_iv_mats(iv_mats = bv_mat, 
                                                                       sector_vec =  p, 
                                                                       input_threshold = c(threshes_l[1], threshes_u[1]), 
                                                                       rm_na_nace = m+1), 
                                                      similarity = "overlap_relative",
                                                      ncores = ncores)


# out vector overlap of firms in sector 3
pw_sim_mat_sector_list_outputs$nace3 
# firm 6 sells only to firms in sector 1 and firm 7 only to firms in sector 5, --> zero overlap






#==============================================================================#
########################### 4. synthetic shocks ################################ 
#==============================================================================#

source(paste0(path, "sample_synthetic_firm_level_shocks.R"))

# firm-level shock to sector 2 (as in Fig. 1b)

# 100% shock to firm 3 in sector 2, 0% to all other firms
xi_0 <- c(rep(0, 2), 1, rep(0, n - 3))

# in- and out-strength of firms
s_in <- colSums(W)
s_out <- rowSums(W)

set.seed(100)
xi_synth <- sample_firm_lev_shocks(psi_k_mat = NULL, # named with nace category, percentage shock to sector k instrength in the first row, and out strength in the second row
                                   firm_lev_shock = xi_0, # n dim vector, elements \in [0,1], empirical shock for each firm
                                   s_in = s_in, # instrengths of firms of sector k 
                                   s_out = s_out, # outstrengths of firms of sector k
                                   n_scen = 10, # number of shocks for the sector,
                                   #m_secs = m_secs, # number of firms within the sector
                                   nace_cat = p, # vector with firms nace categories
                                   tracker = TRUE, 
                                   sample_mode = "empirical",
                                   silent = FALSE)

# shocks only affect firms in sector 2 and they are close to the original shock (i.e., very concentrated to a single firm if possible)
xi_synth$psi_mat




#==============================================================================#
########################### 5. shock propagation ############################### 
#==============================================================================#


#=========================#
###### 5.1 GLcascade ######
#=========================#

## firm-level shock propagation

?GL_cascade
# m times m matrix, where element kl can take values 0 = negligible, 1 = non-essential and 2 = essential, specifies the essentiality of inputs from sector k for the production of a firm in sector l
ess_mat_sec <- matrix(1, m, m, dimnames = list(sectors, sectors)) # i.e., every input for every sector is non-essential


gl_shock_prop_W <- GL_cascade(W = W,
                              p = p,
                              p_sec_impacts = p,   
                              ess_mat_sec =  ess_mat_sec,
                              h_weights = cbind(pmax(s_in, s_out)),       
                              sec_aggr_weights = pmax(s_in, s_out),
                              psi_mat = xi_synth$psi_mat,
                              track_h = TRUE,
                              track_sector_impacts = TRUE, 
                              track_conv = TRUE,
                              conv_type = 1,
                              eps = 10^-2,
                              use_rcpp = FALSE,
                              ncores = 0,
                              run_id = "example_cascade"
)


# losses of individual firms from downstream shock propagation, each scenario is a columns, each row corresponds to the 11 firms
gl_shock_prop_W$hd_T_mat

# network wide losses
gl_shock_prop_W$ESRI

# initial losses of sectors, i.e. 25% shock to sector 2
gl_shock_prop_W$sec_du_mat_init


# indirect losses of sectors
gl_shock_prop_W$sec_du_mat

tot_sector_shocks <- gl_shock_prop_W$sec_du_mat_init + gl_shock_prop_W$sec_du_mat


boxplot(lapply(1:5, function(x) tot_sector_shocks[, x]), ylab = "sector specific losses", "sectors")


## sector level shock propagation 


gl_shock_prop_Z <- GL_cascade(W = Z,
                              p = sectors,
                              ess_mat_sec =  ess_mat_sec,
                              h_weights =  cbind(pmax(colSums(Z), rowSums(Z))),       
                              sec_aggr_weights = pmax(colSums(Z), rowSums(Z)),
                              psi_mat = t(gl_shock_prop_W$sec_du_mat_init), # aggregation of the firm-level shocks
                              track_h = TRUE,
                              track_conv = TRUE,
                              conv_type = 1,
                              eps = 10^-2,
                              use_rcpp = FALSE,
                              ncores = 0,
                              run_id = "example_cascade"
)

# sectors receive idenical shocks for each scenario
gl_shock_prop_Z$hd_T_mat

# production losses of sectors from firm-level shock propagation 
boxplot(lapply(1:5, function(x) tot_sector_shocks[, x]), ylab = "sector specific losses", "sectors")

# production losses of sectors from sector-level shock propagation
points(gl_shock_prop_Z$hd_T_mat[,1], pch = 3, col = "blue", lwd = 2, cex = 3)







#============================#
#### 5.1 Influence Vector ####
#============================#

alpha <- 0.5 # labor share is the dampening factor for pagerank

# pagerank uses out-strength normalization --> transpose W
g_W_transpose <- graph_from_adjacency_matrix(t(W), mode = "directed", weighted = TRUE) 

# influence vector
infl_vec_W <- page_rank(g_W_transpose, damping = 1-alpha, directed = TRUE)$vector

infl_vec_W %*% xi_synth$psi_mat

# economy-wide output losses from firm-level influence vector
hist(as.numeric(infl_vec_W %*% xi_synth$psi_mat), xlab = "economy-wide output losses", main = "influence vector based losses" )


# pagerank uses out-strength normalization --> transpose W
g_Z_transpose <- graph_from_adjacency_matrix(t(Z)) 

# influence vector
infl_vec_Z <- page_rank(g_Z_transpose, damping = 1-alpha, directed = TRUE)$vector

# same result for industry level shocks
infl_vec_Z %*% t(gl_shock_prop_W$sec_du_mat_init)


# economy-wide output losses from firm-level influence vector
hist(as.numeric(infl_vec_W %*% xi_synth$psi_mat), 
     xlab = "economy-wide output losses", main = "influence vector based losses" )

# economy-wide output losses from sector-level influence vector
abline(v = as.numeric(infl_vec_Z %*% t(gl_shock_prop_W$sec_du_mat_init)), 
       lwd = 2, col = "blue", lty = 2)


