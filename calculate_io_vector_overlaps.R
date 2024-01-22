#==============================================================================#
###################### calculate IO-vector overlaps ############################
#==============================================================================#

# Code for calculating IOC and OOC for firms input and output vectors aggregated
# to NACE 2 or 4 digit levels

# 1. functions for aggregating firms' firm-level IO vectors to firms' sector level IO vectors
# 2. functions for calculating various similarity measures pairwise
# 3. functions for calculating various similarlity measures for firms' sector level IO vectors across years


#==============================================================================#
######################## 1. Supply Network Aggregation ############################
#==============================================================================#




# aggregate firm level supply network to sectors on the supplier side
# i.e. suppliers are sectors now and we count the number of suppliers from a sector
# for an unweighted Adjacency Matrix A, or the overall volume for weighted W


# aggregate_suppliers <- function(adj_mat, sector_vec){
#   psup <- Matrix(sapply(sort(unique(sector_vec)), function(k) sector_vec==k))*1
#   
#   list(counts = crossprod(psup, (adj_mat>0)),
#        volume = crossprod(psup, adj_mat)
#   )
# }

aggregate_suppliers <- function(adj_mat, 
                                sector_vec, 
                                unique_sec_vec=NULL){
  if(length(unique_sec_vec)==0){
    psup <- Matrix(sapply(sort(unique(sector_vec)), function(k) sector_vec==k))*1
  }else{
    psup <- Matrix(sapply(unique_sec_vec, function(k) sector_vec==k))*1
  }
  
  list(counts = crossprod(psup, (adj_mat>0)),
       volume = crossprod(psup, adj_mat)
  )
}

# aggregate firm level supply network to sectors on the buyer side
# i.e. buyers are sectors now and we count the number of suppliers / volume from a sector

aggregate_buyers <- function(adj_mat, 
                             sector_vec,
                             unique_sec_vec=NULL){
  
  if(length(unique_sec_vec)==0){
    psup <- Matrix(sapply(sort(unique(sector_vec)), function(k) sector_vec==k))*1
  }else{
    psup <- Matrix(sapply(unique_sec_vec, function(k) sector_vec==k))*1
  }
  
  list(counts = t( (adj_mat>0) %*% psup),
       volume = t(adj_mat %*% psup)
  )
}

# create sector sub matrices containing the input / output vectors of all firms of a
# a sector contained in the sector_vec list
calc_sec_iv_mats <- function(iv_mats, # each column is a firm, each row a sector
                             sector_vec, # gives the sector of each firm 
                             input_threshold=c(0,10^16), # consider firms with overall inputs larger or smaller than threshold
                             rm_na_nace=88){# row number of the NA sector 
  
  uniq_secs <- sort(unique(sector_vec))
  
  sec_iv_mat_list <- as.list(numeric(length(uniq_secs)))
  names(sec_iv_mat_list) <- paste0("nace",uniq_secs)
  
  for(i in 1:length(uniq_secs)){
    
    sector <- uniq_secs[i]
    cat(i, sum(sector_vec==sector) , "  ")
    
    sec_iv_mat_counts <- cbind(iv_mats$counts[-rm_na_nace, sector_vec==sector])
    sec_iv_mat_volume <- cbind(iv_mats$volume[-rm_na_nace, sector_vec==sector])
    
    # consider only firms with min and max number of suppliers
    sec_iv_mat <- cbind(sec_iv_mat_volume[ , (colSums(sec_iv_mat_counts) >= input_threshold[1])&   # lower threshold
                                             (colSums(sec_iv_mat_counts) <= input_threshold[2]) ]) # upper threshold
    
    sec_iv_mat_list[[i]] <- Matrix(sec_iv_mat)
  }
  sec_iv_mat_list[-rm_na_nace]
}


calc_auto_iv_mats <- function(adj_mat_list = A, # list of least length 2, containing supply networks per year, ordered latest year comes first
                              sector_vec_list = p_n4, # list of least length 2, containing industry affiliations
                              p_list = p_list, # contains the ids
                              year = 1,
                              input_threshold = c(0, 10^6),
                              na_sector = "9999",
                              vol_co = "volume",
                              aggregation_direc = "supplier"){ # aggregate suppliers or buyers to sector level
  
  # calculate the firm sectors that are present in all years
  sector_union <- numeric(0)
  for(i in 1:length(sector_vec_list)){
    sector_union <- union(sector_union, sector_vec_list[[i]])
  }
  uni_sec_vec <- sort(sector_union)
  the_na_sector <- which(uni_sec_vec==na_sector)
  
  # aggregate the suppliers to the identified sectors 
  # iv_mat_n4 <- lapply(1:length(sector_vec_list),
  #                     function(x) aggregate_suppliers(adj_mat_list[[x]],
  #                                                     sector_vec_list[[x]],
  #                                                     uni_sec_vec ))
  
  # decide if aggregated over suppliers or buyers
  
  if(aggregation_direc=="supplier"){
    iv_mat_n4 <- list(aggregate_suppliers(adj_mat_list[[year]],
                                          sector_vec_list[[year]],
                                          uni_sec_vec),
                      aggregate_suppliers(adj_mat_list[[year+1]],
                                          sector_vec_list[[year+1]],
                                          uni_sec_vec)
    )
  }
  if(aggregation_direc=="buyer"){
    iv_mat_n4 <- list(aggregate_buyers(adj_mat_list[[year]],
                                       sector_vec_list[[year]],
                                       uni_sec_vec),
                      aggregate_buyers(adj_mat_list[[year+1]],
                                       sector_vec_list[[year+1]],
                                       uni_sec_vec)
    )
  }
  
  #cat(sapply(iv_mat_n4, function(x) dim(x$volume)), "\n")
  
  # calculate the ids of firms that are present in year "year" and "year-1"
  ids <-  intersect(p_list[[year]][,1], p_list[[year+1]][,1])
  
  # subset the aggregated sector times firm matrix to only contain firms appearing in both years
  iv_mat_inters_volume <- list()
  iv_mat_inters_counts <- list()
  
  p_n19_inters <- list()
  
  
  for(j in 1:2){
    select_columns <- p_list[[year+j-1]][ p_list[[year+j-1]][ ,1] %in% ids ,2]
    
    iv_mat_inters_volume[[j]] <- iv_mat_n4[[j]]$volume[ -the_na_sector, # drop the NA row
                                                        select_columns ]
    
    iv_mat_inters_counts[[j]] <- iv_mat_n4[[j]]$counts[ -the_na_sector, # drop the NA row
                                                        select_columns ]
    
    p_n19_inters[[j]] <- sector_vec_list[[year+j-1]][select_columns]
  }
  
  
  cat("all firms \n")
  print(sapply(iv_mat_inters_volume, function(x) dim(x)))
  
  iv_mat_inters_thresh <- list()
  p_n19_inters_thresh <- list()
  
  for(j in 1:2){
    iv_mat_inters_thresh[[j]] <- iv_mat_inters_volume[[j]][ , (colSums(iv_mat_inters_counts[[2]]) >=  input_threshold[1]) & # lower threshold for in/out degree for the earlier year (=year +1)
                                                              (colSums(iv_mat_inters_counts[[2]]) <= input_threshold[2]) ] # upper threshold for in/out degree for the earlier year (=year +1)
    
    p_n19_inters_thresh[[j]] <- p_n19_inters[[j]][ (colSums(iv_mat_inters_counts[[2]]) >=  input_threshold[1]) & # lower threshold for in/out degree
                                                     (colSums(iv_mat_inters_counts[[2]]) <= input_threshold[2]) ] # upper threshold for in/out degree
  }  
  
  cat("firms within the threshold \n")
  print(sapply(iv_mat_inters_thresh, function(x) dim(x)))
  
  list(iv_mats = iv_mat_inters_thresh,
       p_vec = p_n19_inters_thresh[[1]])
}


#==============================================================================#
######################## 2. pairwise IO vector similarities ############################
#==============================================================================#


JaccInd <- function(x,y){
  x <- x>0
  y <- y>0
  
  # intersection divided by union
  overlap <- x%*%y 
  overall <- sum((x|y))
  
  overlap/overall
}



# pairwise Jaccard Index
pairw_ji <- function(mat){ 
  mat <- (mat>0)
  mm <- t(mat)%*%mat
  cs <- colSums(mat)
  mm / (outer(cs, cs, "+")- mm)
}

# relative overlap coefficient
pairw_overlaps_rel <- function(mat){
  mat <- t(t(mat)/colSums(mat))
  apply(mat, 2, function(x) colSums(mat*(mat<=x) + x*(mat>x)))
}


# pairwise jaccard index 
pairw_weigh_ji <- function(mat){
  apply(mat, 2, function(x) colSums(mat*(mat<=x) + x*(mat>x))/
          colSums(x*(mat<=x) + mat*(mat>x)))
}

# relative pairwise jaccard index 
pairw_weigh_ji_rel <- function(mat){
  mat <- t(t(mat)/colSums(mat))
  apply(mat, 2, function(x) colSums(mat*(mat<=x) + x*(mat>x))/
          colSums(x*(mat<=x) + mat*(mat>x)))
}

# cosine similarity 
pairw_cos_sim <- function(mat){
  mm <- t(mat)%*%mat
  cs <- sqrt(colSums( mat^2 ))
  mm / outer(cs, cs)
}

# pairwise link retention rate (fraction of links kept) for temporal comparisons 
pairw_link_retention <- function(mat){ 
  mat <- (mat>0)
  crossprod(mat, mat) / colSums(mat)
}






calc_pw_similarity_mat <- function(mat, # each column is a firm, each row a sector
                                   similarity="jaccard"){# jaccard, weighted_jaccard, overlap, cosine
  
  if(dim(mat)[2]<=1){
    sim_mat <- Matrix(0)
  }else{
    
    if(similarity=="jaccard"){
      sim_mat <- pairw_ji(mat)
      sim_mat <- Matrix(sim_mat*lower.tri(sim_mat))
    }
    if(similarity=="weighted_jaccard"){
      sim_mat <- pairw_weigh_ji(mat)
      sim_mat <- Matrix(sim_mat*lower.tri(sim_mat))
    }
    if(similarity=="weighted_jaccard_relative"){
      sim_mat <- pairw_weigh_ji_rel(mat)
      sim_mat <- Matrix(sim_mat*lower.tri(sim_mat))
    }
    if(similarity=="overlap_relative"){
      sim_mat <- pairw_overlaps_rel(mat)
      sim_mat <- Matrix(sim_mat*lower.tri(sim_mat))
    }
    if(similarity=="cosine"){
      sim_mat <- pairw_cos_sim(mat)
      sim_mat <- Matrix(sim_mat*lower.tri(sim_mat))
    }
    if(similarity=="retention"){
      sim_mat <- pairw_link_retention(mat)
      sim_mat <- Matrix(sim_mat*lower.tri(sim_mat))
    }
    
    
  }
  
  sim_mat
}


calc_pw_sim_mat_list <- function(mat_list, 
                                 similarity = "jaccard",
                                 ncores = detectCores()-2
){
  
  
  cl <- makeCluster(ncores, type = "PSOCK")
  clusterEvalQ(cl, list(library(Matrix)))
  clusterExport(cl, list("pairw_ji", 
                         "pairw_weigh_ji", 
                         "pairw_weigh_ji_rel", 
                         "pairw_overlaps_rel",
                         "pairw_cos_sim",
                         "pairw_link_retention", 
                         "calc_pw_similarity_mat",
                         "similarity" ))
  
  res <- parLapply(cl, mat_list,
                   function(mat) calc_pw_similarity_mat(mat, 
                                                        similarity)
  )
  
  stopCluster(cl)
  
  return(res)
}


#==============================================================================#
######################## 3. timer series IO vector similarities ################
#==============================================================================#


# auto Jaccard Index
auto_ji <- function(mat_t, mat_tm1){ 
  mat_t <- (mat_t>0)
  mat_tm1 <- (mat_tm1>0)
  
  inters_mat <- mat_t * mat_tm1
  union_mat <-  mat_t + mat_tm1 - inters_mat
  
  colSums(inters_mat)/colSums(union_mat)
}

# auto retention
auto_link_retention <- function(mat_t, mat_tm1){ 
  mat_t <- (mat_t > 0)
  mat_tm1 <- (mat_tm1 > 0)
  
  inters_mat <- mat_t * mat_tm1
  
  colSums(inters_mat) / colSums(mat_tm1) # fraction of retained inputs
}

auto_link_retentionII <- function(mat_t, mat_tm1){ 
  
  inters_mat <-  mat_t * (mat_t < mat_tm1) + mat_tm1 * (mat_tm1 <= mat_t)
  
  colSums(inters_mat) / colSums(mat_tm1) # fraction of retained input volume
}

auto_overlaps_rel <- function(mat_t, mat_tm1){ 
  mat_t <- t(t(mat_t)/ifelse(colSums(mat_t)>0, colSums(mat_t), 1))
  mat_tm1 <- t(t(mat_tm1)/ifelse(colSums(mat_tm1)>0, colSums(mat_tm1), 1))
  
  inters_mat <- mat_t * (mat_t < mat_tm1) + mat_tm1 * (mat_tm1<= mat_t)
  colSums(inters_mat)
}

auto_weigh_ji <- function(mat_t, mat_tm1){ 
  
  inters_mat <- mat_t * (mat_t < mat_tm1) + mat_tm1 * (mat_tm1<= mat_t)
  union_mat  <- mat_t * (mat_t > mat_tm1) + mat_tm1 * (mat_tm1>= mat_t)
  
  colSums(inters_mat)/colSums(union_mat)
}

auto_weigh_ji_rel <- function(mat_t, mat_tm1){ 
  mat_t <- t(t(mat_t) / ifelse(colSums(mat_t)>0, colSums(mat_t), 1))
  mat_tm1 <- t(t(mat_tm1) / ifelse(colSums(mat_tm1)>0, colSums(mat_tm1), 1))
  
  inters_mat <- mat_t * (mat_t < mat_tm1) + mat_tm1 * (mat_tm1<= mat_t)
  union_mat  <- mat_t * (mat_t > mat_tm1) + mat_tm1 * (mat_tm1>= mat_t)
  
  colSums(inters_mat) / colSums(union_mat)
}

# cosine similarity 
auto_cos_sim <- function(mat_t, mat_tm1){
  mm <- colSums(mat_t * mat_tm1)
  cs_t <- sqrt(colSums( mat_t^2 ))
  cs_tm1 <- sqrt(colSums( mat_tm1^2 ))
  
  mm / ifelse((cs_t * cs_tm1) > 0, (cs_t * cs_tm1), 1)
}


calc_auto_similarity_mat <- function(mat_t,   # each column is a firm in t and t-1, each row a sector
                                     mat_tm1, # each column is a firm in t and t-1, each row a sector
                                     similarity="jaccard"){# jaccard, weighted_jaccard, overlap, cosine
  
  
  if(similarity=="jaccard"){
    sim_mat <- auto_ji(mat_t, mat_tm1)
  }
  if(similarity=="weighted_jaccard"){
    sim_mat <- auto_weigh_ji(mat_t, mat_tm1)
  }
  if(similarity=="weighted_jaccard_relative"){
    sim_mat <- auto_weigh_ji_rel(mat_t, mat_tm1)
  }
  if(similarity=="overlap_relative"){
    sim_mat <- auto_overlaps_rel(mat_t, mat_tm1)
  }
  if(similarity=="cosine"){
    sim_mat <- auto_cos_sim(mat_t, mat_tm1)
  }
  if(similarity=="retention"){
    sim_mat <- auto_link_retention(mat_t, mat_tm1)
  }
  if(similarity=="retentionII"){
    sim_mat <- auto_link_retentionII(mat_t, mat_tm1)
  }
  
  sim_mat
}
