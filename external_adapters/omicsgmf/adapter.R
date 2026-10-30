args <- commandArgs(trailingOnly=TRUE)

parse_args <- function(args) {
  res <- list(inp=NULL, out=NULL, repo="", config=NULL, seed=42L)
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--in") { res$inp <- args[i+1]; i <- i + 2 }
    else if (args[i] == "--out") { res$out <- args[i+1]; i <- i + 2 }
    else if (args[i] == "--repo") { res$repo <- args[i+1]; i <- i + 2 }
    else if (args[i] == "--config") { res$config <- args[i+1]; i <- i + 2 }
    else if (args[i] == "--seed") { res$seed <- as.integer(args[i+1]); i <- i + 2 }
    else { i <- i + 1 }
  }
  res
}

opts <- parse_args(args)
if (is.null(opts$inp) || is.null(opts$out)) {
  cat("missing --in/--out\n", file=stderr())
  quit(status=2)
}

suppressMessages({
  if (!requireNamespace("omicsGMF", quietly=TRUE)) {
    cat("omicsGMF not installed in this env.\n", file=stderr())
    quit(status=3)
  }
  library(omicsGMF)
})
# Suppress the matrixStats useNames=NA deprecation warning on newer versions
try({
  options(matrixStats.useNames = FALSE)
  matrixStats::matrixStats.options(useNames = FALSE)
}, silent = TRUE)

# Read input
X <- tryCatch({
  as.matrix(utils::read.csv(opts$inp, header=FALSE))
}, error=function(e){
  cat(paste0("read.csv error: ", e$message, "\n"), file=stderr())
  quit(status=2)
})

# Run GMF, impute, and export the protein-level rotation as a sidecar
set.seed(opts$seed)
res <- tryCatch({
  Xt <- t(X)
  dims <- dim(Xt)
  ncomp <- min(16, max(1, min(dims) - 1))
  gmf_fit <- calculateGMF(Xt, family = gaussian(), ncomponents = ncomp)
  imputed <- imputeGMF(Xt, sgdGMF_reducedDims = gmf_fit)
  prot_emb <- tryCatch({
    rot <- attr(gmf_fit, "rotation")
    if (!is.null(rot)) as.matrix(rot) else NULL
  }, error=function(e) NULL)
  list(imputed = t(imputed), protein = prot_emb)
}, error=function(e){
  cat(paste0("calculateGMF/imputeGMF error: ", e$message, "\n"), file=stderr())
  quit(status=2)
})

# Write outputs
tryCatch({
  utils::write.table(res$imputed, file=opts$out, sep=",", row.names=FALSE, col.names=FALSE)
  if (!is.null(res$protein)) {
    prot_out <- paste0(opts$out, ".protein.csv")
    utils::write.table(res$protein, file=prot_out, sep=",", row.names=FALSE, col.names=FALSE)
  }
}, error=function(e){
  cat(paste0("write.table error: ", e$message, "\n"), file=stderr())
  quit(status=2)
})
quit(status=0)


