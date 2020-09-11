#ifndef GRAV3D_H
#define GRAV3D_H

#include<stdio.h>
#include"../global.h"

#if defined TIDES || defined POISSON_TEST
#include <complex>
#endif//TIDES


#ifdef PFFT
#include"potential_PFFT_3D.h"
#endif

#ifdef CUFFT
#include"potential_CUFFT_3D.h"
#endif

#ifdef SOR
#include"potential_SOR_3D.h"
#endif

#ifdef PARIS
#include "potential_paris_3D.h"
#endif

#ifdef HDF5
#include<hdf5.h>
#endif

#define GRAV_ISOLATED_BOUNDARY_X
#define GRAV_ISOLATED_BOUNDARY_Y
#define GRAV_ISOLATED_BOUNDARY_Z

/*! \class Grid3D
 *  \brief Class to create a the gravity object. */
class Grav3D
{
  public:

  Real Lbox_x;
  Real Lbox_y;
  Real Lbox_z;

  Real xMin;
  Real yMin;
  Real zMin;
  /*! \var nx
  *  \brief Total number of cells in the x-dimension */
  int nx_total;
  /*! \var ny
  *  \brief Total number of cells in the y-dimension */
  int ny_total;
  /*! \var nz
  *  \brief Total number of cells in the z-dimension */
  int nz_total;

  /*! \var nx_local
  *  \brief Local number of cells in the x-dimension */
  int nx_local;
  /*! \var ny_local
  *  \brief Local number of cells in the y-dimension */
  int ny_local;
  /*! \var nz_local
  *  \brief Local number of cells in the z-dimension */
  int nz_local;

  /*! \var dx
  *  \brief x-width of cells */
  Real dx;
  /*! \var dy
  *  \brief y-width of cells */
  Real dy;
  /*! \var dz
  *  \brief z-width of cells */
  Real dz;

  #ifdef COSMOLOGY
  Real current_a;
  #endif

  Real dens_avrg ;


  int n_cells;
  int n_cells_potential;


  bool INITIAL;

  Real dt_prev;
  Real dt_now;

  Real Gconst;

  bool TRANSFER_POTENTIAL_BOUNDARIES;
  
  
  bool BC_FLAGS_SET;
  int *boundary_flags;
  
  
  
  #ifdef PFFT
  Potential_PFFT_3D Poisson_solver;
  #endif
  
  #ifdef CUFFT
  Potential_CUFFT_3D Poisson_solver;
  #endif

  #ifdef SOR
  Potential_SOR_3D Poisson_solver;

	#if defined TIDES || defined POISSON_TEST
	std::complex<Real> **Q;
	Real center[3];
	int lmaxBoundaries;
	#endif//TIDES

  #endif//SOR

  #ifdef PARIS
  #if (defined(PFFT) || defined(CUFFT) || defined(SOR))
  #define PARIS_TEST
  Potential_Paris_3D Poisson_solver_test;
  #else
  Potential_Paris_3D Poisson_solver;
  #endif
  #endif


  struct Fields
  {
    /*! \var density
     *  \brief Array containing the density of each cell in the grid */
    Real *density_h;

    /*! \var potential_h
     *  \brief Array containing the gravitational potential of each cell in the grid */
    Real *potential_h;

    /*! \var potential_h
     *  \brief Array containing the gravitational potential of each cell in the grid */
    Real *potential_1_h;
    
    // Arrays for computing the potential values in isolated boundaries
    #ifdef GRAV_ISOLATED_BOUNDARY_X
    Real *pot_boundary_x0;
    Real *pot_boundary_x1;
    #endif
    #ifdef GRAV_ISOLATED_BOUNDARY_Y
    Real *pot_boundary_y0;
    Real *pot_boundary_y1;
    #endif
    #ifdef GRAV_ISOLATED_BOUNDARY_Z
    Real *pot_boundary_z0;
    Real *pot_boundary_z1;
    #endif
  } F;
  
  /*! \fn Grav3D(void)
  *  \brief Constructor for the gravity class */
  Grav3D(void);

  /*! \fn void Initialize(int nx_in, int ny_in, int nz_in)
  *  \brief Initialize the grid. */
  void Initialize( Real x_min, Real y_min, Real z_min, Real Lx, Real Ly, Real Lz, int nx_total, int ny_total, int nz_total, int nx_real, int ny_real, int nz_real, Real dx_real, Real dy_real, Real dz_real, int n_ghost_pot_offset, struct parameters *P);

  void AllocateMemory_CPU(void);
  void Initialize_values_CPU();
  void FreeMemory_CPU(void);
  
  Real Get_Average_Density( );
  Real Get_Average_Density_function( int g_start, int g_end );

  void Set_Boundary_Flags( int *flags );
    
  #ifdef SOR
  void Copy_Isolated_Boundary_To_GPU_buffer( Real *isolated_boundary_h, Real *isolated_boundary_d, int boundary_size );
  void Copy_Isolated_Boundaries_To_GPU( struct parameters *P );
  #endif

	#if defined TIDES || defined POISSON_TEST
	std::complex<Real> Y(int l, int m, Real theta, Real phi);
	#endif

};


#endif //GRAV3D_H
