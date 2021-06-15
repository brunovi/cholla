#if defined(GRAVITY) && defined(PARIS)

#include "potential_paris_3D.h"
#include "gpu.hpp"
#include "../io.h"
#include <cassert>
#include <cfloat>
#include <climits>

static void __attribute__((unused)) prlongDiff(const Real *p, const Real *q, const long ng, const long nx, const long ny, const long nz, const bool plot = false)
{
  Real dMax = 0, dSum = 0, dSum2 = 0;
  Real qMax = 0, qSum = 0, qSum2 = 0;
#pragma omp parallel for reduction(max:dMax,qMax) reduction(+:dSum,dSum2,qSum,qSum2)
  for (long k = 0; k < nz; k++) {
    for (long j = 0; j < ny; j++) {
      for (long i = 0; i < nx; i++) {
        const long ijk = i+ng+(nx+ng+ng)*(j+ng+(ny+ng+ng)*(k+ng));
        const Real qAbs = fabs(q[ijk]);
        qMax = std::max(qMax,qAbs);
        qSum += qAbs;
        qSum2 += qAbs*qAbs;
        const Real d = fabs(q[ijk]-p[ijk]);
        dMax = std::max(dMax,d);
        dSum += d;
        dSum2 += d*d;
      }
    }
  }
  Real maxs[2] = {qMax,dMax};
  Real sums[4] = {qSum,qSum2,dSum,dSum2};
  MPI_Allreduce(MPI_IN_PLACE,&maxs,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&sums,4,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  chprlongf(" Poisson-Solver Diff: L1 %g L2 %g Linf %g\n",sums[2]/sums[0],sqrt(sums[3]/sums[1]),maxs[1]/maxs[0]);
  fflush(stdout);
  if (!plot) return;

  prlongf("###\n");
  const long k = nz/2;
  //for (long j = 0; j < ny; j++) {
  const long j = ny/2;
    for (long i = 0; i < nx; i++) {
      const long ijk = i+ng+(nx+ng+ng)*(j+ng+(ny+ng+ng)*(k+ng));
      //prlongf("%d %d %g %g %g\n",j,i,q[ijk],p[ijk],q[ijk]-p[ijk]);
      prlongf("%d %g %g %g\n",i,q[ijk],p[ijk],q[ijk]-p[ijk]);
    }
    prlongf("\n");
  //}

  MPI_Finalize();
  exit(0);
}

Potential_Paris_3D::Potential_Paris_3D():
  dn_{0,0,0},
  dr_{0,0,0},
  lo_{0,0,0},
  lr_{0,0,0},
  myLo_{0,0,0},
  pp_(nullptr),
  pz_(nullptr),
  minBytes_(0),
  densityBytes_(0),
  potentialBytes_(0),
  da_(nullptr),
  db_(nullptr)
{}

Potential_Paris_3D::~Potential_Paris_3D() { Reset(); }

__device__ static Real analyticD(const Real x, const Real y, const Real z, const Real ddlx, const Real ddly, const Real ddlz)
{
  return exp(-x*x-y*y-z*z)*((4.0*x*x-2.0)*ddlx+(4.0*y*y-2.0)*ddly+(4.0*z*z-2.0)*ddlz);
}

__device__ static Real analyticF(const Real x, const Real y, const Real z)
{
  return exp(-x*x-y*y-z*z);
}

void Potential_Paris_3D::Get_Analytic_Potential(const Real *const density, Real *const potential)
{
  const Real dx = dr_[2];
  const Real dy = dr_[1];
  const Real dz = dr_[0];
  const Real xLo = lo_[2];
  const Real yLo = lo_[1];
  const Real zLo = lo_[0];
  const Real lx = lr_[2];
  const Real ly = lr_[1];
  const Real lz = lr_[0];
  const Real xBegin = myLo_[2];
  const Real yBegin = myLo_[1];
  const Real zBegin = myLo_[0];

  assert(da_);
  Real *const da = da_;
  Real *const db = db_;
  assert(density);
  CHECK(cudaMemcpy(db,density,densityBytes_,cudaMemcpyHostToDevice));

  const Real dlx = 2.0/lx;
  const Real dly = 2.0/ly;
  const Real dlz = 2.0/lz;
  const Real bx = -dlx*(xLo+0.5*lx);
  const Real by = -dly*(yLo+0.5*ly);
  const Real bz = -dlz*(zLo+0.5*lz);
  const Real ddlx = dlx*dlx;
  const Real ddly = dly*dly;
  const Real ddlz = dlz*dlz;

  const long ni = dn_[2];
  const long nj = dn_[1];
  const long nk = dn_[0];

  gpuFor(
    nk,nj,ni,
    GPU_LAMBDA(const long k, const long j, const long i) {
      const Real x = dlx*(xBegin+dx*(Real(i)+0.5))+bx;
      const Real y = dly*(yBegin+dy*(Real(j)+0.5))+by;
      const Real z = dlz*(zBegin+dz*(Real(k)+0.5))+bz;
      const long iab = i+ni*(j+nj*k);
      da[iab] = db[iab]-analyticD(x,y,z,ddlx,ddly,ddlz);
    });

  assert(pz_);
  pz_->solve(minBytes_,da,db);

  const long ngi = ni+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long ngj = nj+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;

  gpuFor(
    nk,nj,ni,
    GPU_LAMBDA(const long k, const long j, const long i) {
      const Real x = dlx*(xBegin+dx*(Real(i)+0.5))+bx;
      const Real y = dly*(yBegin+dy*(Real(j)+0.5))+by;
      const Real z = dlz*(zBegin+dz*(Real(k)+0.5))+bz;
      const long ia = i+ni*(j+nj*k);
      const long ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
      db[ib] = da[ia]+analyticF(x,y,z);
    });

  assert(potential);
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToHost));
}

void Potential_Paris_3D::Get_Potential(const Real *const density, Real *const potential, const Real g, const Real offset, const Real a)
{
#ifdef COSMOLOGY
  const Real scale = Real(4)*M_PI*g/a;
#else
  const Real scale = Real(4)*M_PI*g;
#endif
  assert(da_);
  Real *const da = da_;
  Real *const db = db_;
  assert(density);

  const long ni = dn_[2];
  const long nj = dn_[1];
  const long nk = dn_[0];

  const long n = ni*nj*nk;
  #ifdef GRAVITY_GPU
  CHECK(cudaMemcpy(db,density,densityBytes_,cudaMemcpyDeviceToDevice));
  #else
  CHECK(cudaMemcpy(db,density,densityBytes_,cudaMemcpyHostToDevice));
  #endif
  const long ngi = ni+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long ngj = nj+N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;

  if (pp_) {

    gpuFor(n,GPU_LAMBDA(const long i) { db[i] = scale*(db[i]-offset); });
    pp_->solve(minBytes_,db,da);
    gpuFor(
      nk,nj,ni,
      GPU_LAMBDA(const long k, const long j, const long i) {
        const long ia = i+ni*(j+nj*k);
        const long ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
        db[ib] = da[ia];
      });

  } else {

    assert(pz_);
    constexpr Real fraction = 2*6;
    const Real r0 = std::min(std::min(lr_[0],lr_[1]),lr_[2])/fraction;
    const Real spi = sqrt(M_PI);
    const Real denom = r0*spi;
    const Real mass = a;
    const Real rho0 = mass/(denom*denom*denom);

    const Real dlx = lr_[2]/r0;
    const Real dly = lr_[1]/r0;
    const Real dlz = lr_[0]/r0;
    const Real dx = dr_[2]*dlx;
    const Real dy = dr_[1]*dly;
    const Real dz = dr_[0]*dlz;
    const Real x0 = lo_[2]+0.5*lr_[2];
    const Real y0 = lo_[1]+0.5*lr_[1];
    const Real z0 = lo_[0]+0.5*lr_[0];
    const Real xBegin = (myLo_[2]-x0)*dlx;
    const Real yBegin = (myLo_[1]-y0)*dly;
    const Real zBegin = (myLo_[0]-z0)*dlz;

    gpuFor(
      nk,nj,ni,
      GPU_LAMBDA(const long k, const long j, const long i) {
        const Real x = xBegin+dx*(Real(i)+0.5);
        const Real y = yBegin+dy*(Real(j)+0.5);
        const Real z = zBegin+dz*(Real(k)+0.5);
        const long iab = i+ni*(j+nj*k);
        da[iab] = scale*(db[iab]-offset-rho0*exp(-x*x-y*y-z*z));
      });

    pz_->solve(minBytes_,da,db);

    const Real ngmdr0 = -g*mass/r0;
    const Real lim0 = ngmdr0*Real(2)/spi;
    const Real lim2 = -lim0/Real(3);

    gpuFor(
      nk,nj,ni,
      GPU_LAMBDA(const long k, const long j, const long i) {
        const Real x = xBegin+dx*(Real(i)+0.5);
        const Real y = yBegin+dy*(Real(j)+0.5);
        const Real z = zBegin+dz*(Real(k)+0.5);
        const Real r = sqrt(x*x+y*y+z*z);
        const Real v0 = (r < DBL_EPSILON) ? lim0+lim2*r*r : ngmdr0*erf(r)/r;
        const long ia = i+ni*(j+nj*k);
        const long ib = i+N_GHOST_POTENTIAL+ngi*(j+N_GHOST_POTENTIAL+ngj*(k+N_GHOST_POTENTIAL));
        db[ib] = da[ia]+v0;
      });
  }
  assert(potential);
  #ifdef GRAVITY_GPU
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToDevice));
  #else
  CHECK(cudaMemcpy(potential,db,potentialBytes_,cudaMemcpyDeviceToHost));
  #endif
}

void Potential_Paris_3D::Initialize(const Real lx, const Real ly, const Real lz, const Real xMin, const Real yMin, const Real zMin, const int nx, const int ny, const int nz, const int nxReal, const int nyReal, const int nzReal, const Real dx, const Real dy, const Real dz, const bool periodic)
{
  chprlongf(" Using Poisson Solver: Paris ");
  if (periodic) chprlongf("Periodic");
  else chprlongf("Antisymmetric");
#ifdef PARIS_5PT
  chprlongf(" 5-Polong\n");
#elif defined PARIS_3PT
  chprlongf(" 3-Polong\n");
#else
  chprlongf(" Spectral\n");
#endif

  const long nl012 = long(nxReal)*long(nyReal)*long(nzReal);
  assert(nl012 <= long_MAX);

  dn_[0] = nzReal;
  dn_[1] = nyReal;
  dn_[2] = nxReal;

  dr_[0] = dz;
  dr_[1] = dy;
  dr_[2] = dx;

  lr_[0] = lz;
  lr_[1] = ly;
  lr_[2] = lx;

  myLo_[0] = zMin;
  myLo_[1] = yMin;
  myLo_[2] = xMin;
  MPI_Allreduce(myLo_,lo_,3,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);

  const Real hi[3] = {lo_[0]+lz-dr_[0],lo_[1]+ly-dr_[1],lo_[2]+lx-dr_[2]};
  const long n[3] = {nz,ny,nx};
  const long m[3] = {n[0]/nzReal,n[1]/nyReal,n[2]/nxReal};
  const long id[3] = {long(round((zMin-lo_[0])/(dn_[0]*dr_[0]))),long(round((yMin-lo_[1])/(dn_[1]*dr_[1]))),long(round((xMin-lo_[2])/(dn_[2]*dr_[2])))};
  chprlongf("  Paris: [ %g %g %g ]-[ %g %g %g ] N_local[ %d %d %d ] Tasks[ %d %d %d ]\n",lo_[2],lo_[1],lo_[0],lo_[2]+lx,lo_[1]+ly,lo_[0]+lz,dn_[2],dn_[1],dn_[0],m[2],m[1],m[0]);

  assert(dn_[0] == n[0]/m[0]);
  assert(dn_[1] == n[1]/m[1]);
  assert(dn_[2] == n[2]/m[2]);

  if (periodic) {
    pp_ = new PoissonPeriodic3x1DBlockedGPU(n,lo_,hi,m,id);
    assert(pp_);
    minBytes_ = pp_->bytes();
  } else {
    pz_ = new PoissonZero3DBlockedGPU(n,lo_,hi,m,id);
    assert(pz_);
    minBytes_ = pz_->bytes();
 }

  densityBytes_ = long(sizeof(Real))*dn_[0]*dn_[1]*dn_[2];
  const long gg = N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  potentialBytes_ = long(sizeof(Real))*(dn_[0]+gg)*(dn_[1]+gg)*(dn_[2]+gg);

  CHECK(cudaMalloc(relongerpret_cast<void **>(&da_),std::max(minBytes_,densityBytes_)));
  assert(da_);
  
  CHECK(cudaMalloc(relongerpret_cast<void **>(&db_),std::max(minBytes_,potentialBytes_)));
  assert(db_);
}

void Potential_Paris_3D::Reset()
{
  if (db_) CHECK(cudaFree(db_));
  db_ = nullptr;

  if (da_) CHECK(cudaFree(da_));
  da_ = nullptr;

  potentialBytes_ = densityBytes_ = minBytes_ = 0;

  if (pz_) delete pz_;
  pz_ = nullptr;

  if (pp_) delete pp_;
  pp_ = nullptr;

  myLo_[2] = myLo_[1] = myLo_[0] = 0;
  lr_[2] = lr_[1] = lr_[0] = 0;
  lo_[2] = lo_[1] = lo_[0] = 0;
  dr_[2] = dr_[1] = dr_[0] = 0;
  dn_[2] = dn_[1] = dn_[0] = 0;
}

#endif
