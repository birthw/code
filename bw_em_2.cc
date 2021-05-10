#include "bazaar/FileUtil.hh"
#include "bazaar/MarkerMap.hh"
#include "LRP/SharingData.hh"
#include "chi/chiReader.hpp"
#include "bazaar/debug.hh"
#include "bazaar/POptions.hh"

#include <bfgs/bfgs.h>
#include <bfgs/numgrad.h>

#include "assoc_util.hh"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <mkl.h>

#define MATHLIB_STANDALONE 1
#include "Rmath.h"

class BW_EM : public POptions {
public:
  BW_EM() :
      POptions("bw_em call synopsis:\n\n>bw_em [options] <proband-mother-file> <proband-bw-file> <chi-files>\n\nAllowed options:") {
    m_options.add_options()
      ("debug_output_level,d", po::value<unsigned int>()->default_value(0), "Debug output level. 0 means no output")
      ("markers", po::value<std::string>(), "Only print results for these markers")
      ("linkage_data", po::value<std::string>(), "Linkage data")
      ("interval,i", po::value<std::string>(), "Test for association within interval only")
      ("info_threshold", po::value<double>()->default_value(.0), "Only perform analysis for markers with imputation info above this threshold")
      ("threads", po::value<size_t>(), "The number of threads available")
      ;
  }
};

double estimate_mat_nt(double proband_mat, double mat_1, double mat_2, double freq) {
  const double mat_gen = (mat_1 < 0 ? freq : mat_1) + (mat_2 < 0 ? freq : mat_2);
  const double mat_nt = mat_gen - proband_mat;
  assert(!isnan(mat_nt));
  if (mat_nt < .0)
    return .0;
  else if (mat_nt > 1.)
    return 1.;
  else
    return mat_nt;
}

class Expectation_G {
public:
  double EP;
  double EM;
  double EN;

  double EPM;
  double EPN;
  double EMN;

  Expectation_G(double EP0=NAN, double EM0=NAN, double EN0=NAN) :
      EP(EP0), EM(EM0), EN(EN0), EPM(NAN), EPN(NAN), EMN(NAN) {
  }
};

double expectation(std::vector<Expectation_G> &EG, const std::vector<double> &bw,
                   double *proband_pat, double *proband_mat, double *proband_mat_nt,
                   double alpha, double beta_P, double beta_M, double beta_N, double sd) {
  double loglik = .0;
  EG.resize(bw.size());
  for (size_t i = 0; i < bw.size(); i++) {
    double prob_nc = .0;
    double prob_P = .0;
    double prob_M = .0;
    double prob_N = .0;
    double prob_PM = .0;
    double prob_PN = .0;
    double prob_MN = .0;
    for (size_t P = 0; P < 2; P++) {
      const double xb_1 = alpha + P*beta_P;
      const double prob_1 = P == 0 ? 1. - proband_pat[i] : proband_pat[i];
      if (prob_1 != 0) {
        for (size_t M = 0; M < 2; M++) {
          const double xb_2 = xb_1 + M*beta_M;
          const double prob_2 = prob_1*(M == 0 ? 1. - proband_mat[i] : proband_mat[i]);
          if (prob_2 != 0) {
            for (size_t N = 0; N < 2; N++) {
              const double xb_3 = xb_2 + N*beta_N;
              const double prob_3 = prob_2*(N == 0 ? 1. - proband_mat_nt[i] : proband_mat_nt[i])*dnorm(bw[i], xb_3, sd, 0);
              if (prob_3 != 0) {
                prob_nc += prob_3;
                if (P == 1)
                  prob_P += prob_3;
                if (M == 1)
                  prob_M += prob_3;
                if (N == 1)
                  prob_N += prob_3;
                if (P == 1 and M == 1)
                  prob_PM += prob_3;
                if (P == 1 and N == 1)
                  prob_PN += prob_3;
                if (M == 1 and N == 1)
                  prob_MN += prob_3;
              }
            }
          }
        }
      }
    }
    assert(prob_nc > .0);
    EG[i].EP = prob_P/prob_nc;
    assert(not isnan(EG[i].EP));
    EG[i].EM = prob_M/prob_nc;
    EG[i].EN = prob_N/prob_nc;
    EG[i].EPM = prob_PM/prob_nc;
    EG[i].EPN = prob_PN/prob_nc;
    EG[i].EMN = prob_MN/prob_nc;
    loglik += log(prob_nc);
  }

  return loglik;
}

class Model {
public:
  class Parameter {
  public:
    std::vector<bool> use;
    Parameter(bool use_P, bool use_M, bool use_N) : use(3, false) {
      if (use_P)
        use[0] = true;
      if (use_M)
        use[1] = true;
      if (use_N)
        use[2] = true;
    }

  };

  std::vector<Parameter> parameters;

  Model() {}

  void add_parameter(bool use_P, bool use_M, bool use_N) {
    parameters.push_back(Parameter(use_P, use_M, use_N));
  }

};

double LL(const std::vector<double> &bw,
          double *proband_pat, double *proband_mat, double *proband_mat_nt,
          double alpha, double beta_P, double beta_M, double beta_N, double sd) {
  double loglik = .0;
  for (size_t i = 0; i < bw.size(); i++) {
    double prob_nc = .0;
    for (size_t P = 0; P < 2; P++) {
      const double xb_1 = alpha + P*beta_P;
      const double prob_1 = P == 0 ? 1. - proband_pat[i] : proband_pat[i];
      if (prob_1 != 0) {
        for (size_t M = 0; M < 2; M++) {
          const double xb_2 = xb_1 + M*beta_M;
          const double prob_2 = prob_1*(M == 0 ? 1. - proband_mat[i] : proband_mat[i]);
          if (prob_2 != 0) {
            for (size_t N = 0; N < 2; N++) {
              const double xb_3 = xb_2 + N*beta_N;
              const double prob_3 = prob_2*(N == 0 ? 1. - proband_mat_nt[i] : proband_mat_nt[i])*dnorm(bw[i], xb_3, sd, 0);
              prob_nc += prob_3;
            }
          }
        }
      }
    }
    loglik += isnan(prob_nc) or prob_nc == .0 ? -1e300 : log(prob_nc);
    assert(not isnan(loglik));
  }
  return loglik;
}

class NegLL {
private:
  const Model mod;
  const std::vector<double> &bw;
  double *proband_pat;
  double *proband_mat;
  double *proband_mat_nt;

public:
  NegLL(const Model &m, const std::vector<double> &y, double *pat, double *mat, double *mat_nt) :
      mod(m), bw(y), proband_pat(pat), proband_mat(mat), proband_mat_nt(mat_nt) {}

  double operator()(const std::vector<double> &beta) {
    const double alpha = beta[0];
    double beta_P = .0;
    double beta_M = .0;
    double beta_N = .0;
    for (size_t p = 0; p < mod.parameters.size(); p++) {
      if (mod.parameters[p].use[0])
        beta_P += beta[p + 1];
      if (mod.parameters[p].use[1])
        beta_M += beta[p + 1];
      if (mod.parameters[p].use[2])
        beta_N += beta[p + 1];
    }
    const double sd = fabs(beta[beta.size() - 1]);
    const double ll = LL(bw, proband_pat, proband_mat, proband_mat_nt, alpha, beta_P, beta_M, beta_N, sd > 10. ? 10. : sd);
    if (debug_output_level > 0)
      std::cerr << alpha << '\t' << beta_P << '\t' << beta_M << '\t' << beta_N << '\t' << sd << '\t' << ll << std::endl;
    return -ll;
  }

};

class EM {
private:
  double e_step(double *Egy, double *Egg, double *beta, double sd, const Model &mod, const std::vector<double> &bw,
                double *proband_pat, double *proband_mat, double *proband_mat_nt) {
    const double alpha = beta[0];
    double beta_P = .0;
    double beta_M = .0;
    double beta_N = .0;
    for (size_t p = 0; p < mod.parameters.size(); p++) {
      if (mod.parameters[p].use[0])
        beta_P += beta[p + 1];
      if (mod.parameters[p].use[1])
        beta_M += beta[p + 1];
      if (mod.parameters[p].use[2])
        beta_N += beta[p + 1];
    }
    std::vector<Expectation_G> EG;
    double loglik = expectation(EG, bw, proband_pat, proband_mat, proband_mat_nt, alpha, beta_P, beta_M, beta_N, sd);
    bzero(Egy, sizeof(double)*4);
    bzero(Egg, sizeof(double)*4*4);
    for (size_t i = 0; i < bw.size(); i++) {
      Egy[0] += bw[i];
      assert(not isnan(EG[i].EP));
      Egy[1] += EG[i].EP*bw[i];
      assert(not isnan(Egy[1]));
      Egy[2] += EG[i].EM*bw[i];
      Egy[3] += EG[i].EN*bw[i];

      Egg[0] += 1.;

      Egg[1] += EG[i].EP;
      Egg[4] += EG[i].EP;
      Egg[4 + 1] += EG[i].EP;

      Egg[2] += EG[i].EM;
      Egg[2*4] += EG[i].EM;
      Egg[2*4 + 2] += EG[i].EM;

      Egg[3] += EG[i].EN;
      Egg[3*4] += EG[i].EN;
      Egg[3*4 + 3] += EG[i].EN;

      Egg[4 + 2] += EG[i].EPM;
      Egg[2*4 + 1] += EG[i].EPM;

      Egg[4 + 3] += EG[i].EPN;
      Egg[3*4 + 1] += EG[i].EPN;

      Egg[2*4 + 3] += EG[i].EMN;
      Egg[3*4 + 2] += EG[i].EMN;
    }

    return loglik;
  }

  void m_step(double *beta, double &sd, double *Q, double *Egy, double *Egg, double Syy, size_t num_param, size_t num) {
//    const char no = 'n';
//    const char yes = 'y';
    // calculate Q Egy and store in QEgy
    double *QEgy = new double[num_param];
    cblas_dgemv(CblasColMajor, CblasNoTrans, num_param, 4, 1., Q, num_param, Egy, 1, .0, QEgy, 1);

    // calculate Egg Q^T and store in EggQt
    double *EggQt = new double[4*num_param];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 4, num_param, 4, 1., Egg, 4, Q, num_param, .0, EggQt, 4);

    // calculate Q Egg Q^T and store in QEggQt
    double *QEggQt = new double[num_param*num_param];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, num_param, num_param, 4, 1., Q, num_param, EggQt, 4, .0, QEggQt, num_param);

    // Solve Q Egg Q^T beta = Q Egy for beta
    LAPACKE_dposv(LAPACK_COL_MAJOR, 'U', num_param, 1, QEggQt, num_param, QEgy, num_param);
    memcpy(beta, QEgy, sizeof(double)*num_param);

    // Estimate sd
    const double dp = cblas_ddot(num_param, beta, 1, Egy, 1);
    const double var = 1./double(num)*(Syy - dp);
    sd = sqrt(var < 0 ? Syy/num : var);

    delete [] QEgy;
    delete [] EggQt;
    delete [] QEggQt;
  }

public:
  EM(const Model &mod, const std::vector<double> &bw, double *proband_pat, double *proband_mat, double *proband_mat_nt) : model(mod) {
    const double TOL = 1e-3;
    const size_t min_iter = 10;
    const size_t max_iter = 1000;

    double Syy = 0;
    for (size_t i = 0; i <= bw.size(); i++)
      Syy += square(bw[i]);

    const size_t num_param = 1 + model.parameters.size();
    beta = new double[num_param];
    bzero(beta, sizeof(double)*num_param);
    double sd = 1.;
    double *Egy = new double[4];
    double *Egg = new double[4*4];

    double *Q = new double[num_param*4];
    bzero(Q, sizeof(double)*num_param*4);
    Q[0] = 1.;
    for (size_t p = 1; p < num_param; p++)
      for (size_t i = 1; i < 4; i++)
        Q[i*num_param + p] = model.parameters[p - 1].use[i - 1] ? 1. : .0;

    loglik = NAN;
    double loglik_last = NAN;
    for (size_t iter = 0; iter < max_iter; iter++) {
      // E step
      loglik = e_step(Egy, Egg, beta, sd, mod, bw, proband_pat, proband_mat, proband_mat_nt);

      if (debug_output_level > 0) {
        std::cerr << iter << '\t' << loglik << '\t' << sd;
        for (size_t i = 0; i < num_param; i++)
          std::cerr << '\t' << beta[i];
        std::cerr << std::endl;
      }

      // M step
      m_step(beta, sd, Q, Egy, Egg, Syy, num_param, bw.size());

      // check for convergence
      if (iter > min_iter and fabs(loglik - loglik_last) < TOL)
        break;
      loglik_last = loglik;

      if (iter + 1 == max_iter)
        std::cerr << "Warning: maximum number of iterations reached in EM algorithm" << std::endl;	
    }

    delete [] Egy;
    delete [] Egg;
    delete [] Q;
  }

  Model model;
  double loglik;
  double *beta;

};

void bw_tests(const std::vector<double> &bw, double *proband_pat, double *proband_mat, double *proband_mat_nt) {
  // Null model
  Model model_null;

  // Child effect model
  Model model_child;
  model_child.add_parameter(true, true, false);

  // Paternal effect model
  Model model_paternal;
  model_paternal.add_parameter(true, false, false);

  // Maternal effect model
  Model model_maternal;
  model_maternal.add_parameter(false, true, false);

  // Mother effect model
  Model model_mother;
  model_mother.add_parameter(false, true, true);

  // Child, maternal effect model
  Model model_child_mother;
  model_child_mother.add_parameter(false, true, true);
  model_child_mother.add_parameter(true, true, false);

  // maternal, non-trans effect model
  Model model_maternal_non_trans;
  model_maternal_non_trans.add_parameter(false, true, false);
  model_maternal_non_trans.add_parameter(false, false, true);

  // paternal, non-trans effect model
  Model model_paternal_non_trans;
  model_paternal_non_trans.add_parameter(true, false, false);
  model_paternal_non_trans.add_parameter(false, false, true);

  // paternal, maternal effect model
  Model model_paternal_maternal;
  model_paternal_maternal.add_parameter(true, false, false);
  model_paternal_maternal.add_parameter(false, true, false);

  // paternal, mother effect model
  Model model_par;
  model_par.add_parameter(true, false, false);
  model_par.add_parameter(false, true, true);

  // full model
  Model model_full;
  model_full.add_parameter(true, false, false);
  model_full.add_parameter(false, true, false);
  model_full.add_parameter(false, false, true);

  EM em_null(model_null, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_child(model_child, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_paternal(model_paternal, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_maternal(model_maternal, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_mother(model_mother, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_child_mother(model_child_mother, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_maternal_non_trans(model_maternal_non_trans, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_paternal_non_trans(model_paternal_non_trans, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_paternal_maternal(model_paternal_maternal, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_par(model_par, bw, proband_pat, proband_mat, proband_mat_nt);
  EM em_full(model_full, bw, proband_pat, proband_mat, proband_mat_nt);

  std::cout << '\t' << em_child.beta[1] << '\t' << 2*(em_child.loglik - em_null.loglik)
            << '\t' << em_paternal.beta[1] << '\t' << 2*(em_paternal.loglik - em_null.loglik)
            << '\t' << em_maternal.beta[1] << '\t' << 2*(em_maternal.loglik - em_null.loglik)
            << '\t' << em_mother.beta[1] << '\t' << 2*(em_mother.loglik - em_null.loglik)
            << '\t' << em_child_mother.beta[1] << '\t' << em_child_mother.beta[2] << '\t' << 2*(em_child_mother.loglik - em_null.loglik)
            << '\t' << 2*(em_child_mother.loglik - em_child.loglik) << '\t' << 2*(em_child_mother.loglik - em_mother.loglik)
            << '\t' << em_paternal_maternal.beta[1] << '\t' << em_paternal_maternal.beta[2] << '\t' << 2*(em_paternal_maternal.loglik - em_null.loglik)
            << '\t' << 2*(em_paternal_maternal.loglik - em_child.loglik)
            << '\t' << em_par.beta[1] << '\t' << em_par.beta[2] << '\t' << 2*(em_par.loglik - em_null.loglik)
            << '\t' << em_full.beta[1] << '\t' << em_full.beta[2]<< '\t' << em_full.beta[3]
            << '\t' << 2*(em_full.loglik - em_maternal_non_trans.loglik)
            << '\t' << 2*(em_full.loglik - em_paternal_non_trans.loglik)
            << '\t' << 2*(em_full.loglik - em_paternal_maternal.loglik)
            << '\t' << 2*(em_full.loglik - em_par.loglik);

}

int main(int argc, char *argv[]) {
  BW_EM opts;

  opts.parse_commandline(argc, argv);

  const std::vector<std::string> arguments = opts.arguments(); // must use opts.arguments() like this!

  if (opts.printHelp() or arguments.size() < 3) {
    std::cout << opts.description() << "\n";
    return 1;
  }

  debug_output_level = opts.get<unsigned int>("debug_output_level");

  const double info_threshold = opts.get<double>("info_threshold");

  if (opts.contains("threads"))
    omp_set_num_threads(opts.get<size_t>("threads"));

  //------------------------------------------------------------------
  // Read proband data
  std::map<std::string, std::string> proband_mother;
  std::map<std::string, double> proband_bw;
  read_map(arguments[0], proband_mother);
  read_map(arguments[1], proband_bw);
  std::set<std::string> phenotyped_pns;
  for (auto &i: proband_mother) {
    phenotyped_pns.insert(i.first);
    if (i.second != "" and i.second != "0")
      phenotyped_pns.insert(i.second);
  }

  //------------------------------------------------------------------
  // Open genotype input files
  if (debug_output_level > 0)
    std::cerr << "Reading genotype header files" << std::endl;
  std::vector<chiReader*> genotype_files;
  std::map<std::string, size_t> gt_idx;
  open_chi_files(genotype_files, gt_idx, arguments.begin() + 2, arguments.end());

  //----------------------------------------------------------------------
  // Read which markers to analyse
  std::set<std::string> markers;
  if (opts.contains("markers"))
    read_file(markers, opts.get<std::string>("markers"));

  //----------------------------------------------------------------------
  // Read PNs with linkage information
  LinkageData2 *lin_data = 0;
  LinkageImputer2 *lin = 0;
  size_t num_gts;
  num_gts = setup_linkage_data(lin_data, lin, opts, genotype_files, &phenotyped_pns, gt_idx, markers);
  assert(num_gts > 0);

  //----------------------------------------------------------------------
  // Find which PNs have QT data
  std::map<std::string, size_t> pn_qt_idx;
  for (auto &i: proband_bw)
    if (gt_idx.find(i.first) != gt_idx.end() and pn_qt_idx.find(i.first) == pn_qt_idx.end()) {
      const size_t idx = pn_qt_idx.size();
      pn_qt_idx[i.first] = idx;
    }
  const size_t N = pn_qt_idx.size();
  if (N < 2)
    throw error(std::string("only ") + int(N) + " PNs with phenotype data!");
  std::vector<size_t> proband_gt_idx(N);
  std::vector<size_t> mother_gt_idx(N);
  std::vector<double> bw(N);
  for (auto &i: pn_qt_idx) {
    proband_gt_idx[i.second] = gt_idx[i.first];
    assert(proband_mother.find(i.first) != proband_mother.end());
    std::string mother = proband_mother[i.first];
    if (mother == "" or mother == "0" or gt_idx.find(mother) == gt_idx.end())
      mother_gt_idx[i.second] = size_t(-1);
    else
      mother_gt_idx[i.second] = gt_idx[mother];
    bw[i.second] = proband_bw[i.first];
  }

  //------------------------------------------------------------------
  // Print header
  std::cout << "Marker\tbeta_child\tX2_child\tbeta_paternal\tX2_paternal\tbeta_maternal\tX2_maternal\tbeta_mother\tX2_mother\tbeta_cm_child\tbeta_cm_mother\tX2_cm_vs_null\tX2_cm_vs_child\t\tX2_cm_vs_mother\tbeta_pm_paternal\tbeta_pm_maternal\tX2_pm_vs_null\tX2_pm_vs_child\tbeta_par_mother\tbeta_par_paternal\tx2_beta_par_vs_null\tbeta_full_paternal\tbeta_full_maternal\tbeta_full_non_trans\tx2_full_vs_no_paternal\tx2_full_vs_no_maternal\tx2_full_vs_no_non_trans\tx2_full_vs_par\n";

  //------------------------------------------------------------------
  // Read genotypes and perform analysis
  float *v = new float[2*num_gts];
  double *proband_pat = new double[N];
  double *proband_mat = new double[N];
  double *proband_mat_nt = new double[N];
  for (auto &gf: genotype_files) {
    if (lin != 0)
      lin->reset();
    gf->v = v;
    std::fill(gf->v, gf->v + 2*num_gts, float(-1.));
    for (size_t midx = 0; midx < gf->markers.size(); midx++) {
      if ((not markers.empty() and markers.count(gf->markers[midx].name) == 0))
        continue;

      const float iv = info_threshold > 0 ? gf->markers[midx].info : 1.;
      const double freq = gf->markers[midx].freq;
      if (isnan(freq) or freq <= .0 or freq >= 1. or isnan(iv) or iv < info_threshold)
        continue;

      gf->seek(midx);
      gf->read_next_marker();
      if (lin != 0)
        lin->impute(gf->v, gf->markers[midx].freq);

      for (size_t i = 0; i < N; i++) {
        const size_t pi = proband_gt_idx[i];
        proband_pat[i] = v[2*pi];
        if (proband_pat[i] == -1)
          proband_pat[i] = freq;
        proband_mat[i] = v[2*pi + 1];
        if (proband_mat[i] == -1)
          proband_mat[i] = freq;
        const size_t mi = mother_gt_idx[i];
        if (mi == size_t(-1))
          proband_mat_nt[i] = freq;
        else
          proband_mat_nt[i] = estimate_mat_nt(proband_mat[i], v[2*mi], v[2*mi + 1], freq);
      }

      std::cout << gf->markers[midx].name;
      bw_tests(bw, proband_pat, proband_mat, proband_mat_nt);
      std::cout << std::endl;
    }
  }

  return 0;
}
