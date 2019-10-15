#ifdef PARTICLES
#include <unistd.h>
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include"../global.h"
#include"../grid3D.h"
#include"../io.h"
#include"particles_3D.h"

#ifdef HDF5
#include<hdf5.h>
#endif
#ifdef MPI_CHOLLA
#include"../mpi_routines.h"
#endif
#include <unistd.h>


void Particles_3D::Load_Particles_Data( struct parameters *P, struct Header &H){
  char filename[100];
  char timestep[20];
  int nfile = P->nfile; //output step you want to read from
  char filename_counter[100];
  // create the filename to read from

  strcpy(filename, P->indir);
  sprintf(timestep, "%d_particles", nfile);
  strcat(filename,timestep);

  #if defined BINARY
  chprintf("\nERROR: Particles only support HDF5 outputs\n")
  #elif defined HDF5
  strcat(filename,".h5");
  #endif

  #ifdef MPI_CHOLLA
  #ifdef TILED_INITIAL_CONDITIONS
  sprintf(filename,"%s",filename); //Everyone reads from the same file
  #else
  sprintf(filename,"%s.%d",filename,procID);
  #endif //TILED_INITIAL_CONDITIONS
  #endif

  chprintf(" Loading particles file: %s \n", filename );

  #ifdef HDF5
  hid_t  file_id;
  herr_t  status;

  // open the file
  file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    printf("Unable to open input file.\n");
    exit(0);
  }

  Load_Particles_Data_HDF5(file_id, nfile, P, H );

  #endif
}


void Grid3D::WriteData_Particles( struct parameters P, int nfile)
{
  // Write the particles data to file
  OutputData_Particles( P, nfile);
}


#ifdef HDF5

void Particles_3D::Load_Particles_Data_HDF5(hid_t file_id, int nfile, struct parameters *P, struct Header &H  )
{
  int i, j, k, id, buf_id;
  hid_t     attribute_id, dataset_id;
  Real      *dataset_buffer_px;
  Real      *dataset_buffer_py;
  Real      *dataset_buffer_pz;
  Real      *dataset_buffer_vx;
  Real      *dataset_buffer_vy;
  Real      *dataset_buffer_vz;
  Real      *dataset_buffer_m;
  herr_t    status;

  part_int_t n_to_load, pIndx;

  attribute_id = H5Aopen(file_id, "n_particles_local", H5P_DEFAULT);
  status = H5Aread(attribute_id, H5T_NATIVE_LONG, &n_to_load);
  status = H5Aclose(attribute_id);

  #ifdef COSMOLOGY
  attribute_id = H5Aopen(file_id, "current_z", H5P_DEFAULT);
  status = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &current_z);
  status = H5Aclose(attribute_id);

  attribute_id = H5Aopen(file_id, "current_a", H5P_DEFAULT);
  status = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &current_a);
  status = H5Aclose(attribute_id);
  #endif

  #ifdef SINGLE_PARTICLE_MASS
  attribute_id = H5Aopen(file_id, "particle_mass", H5P_DEFAULT);
  status = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &particle_mass);
  status = H5Aclose(attribute_id);
  chprintf( " Using Single mass for DM particles: %f  Msun/h\n", particle_mass);
  #endif

  #ifndef MPI_CHOLLA
  // if ( n_total != G.Grav.nx_total * G.Grav.ny_total * G.Grav.nz_total) break;
  chprintf(" Loading %ld particles\n", n_to_load);
  // #endif
  #else
  part_int_t n_total_load;
  n_total_load = ReducePartIntSum( n_to_load );
  chprintf( " Total Particles To Load: %ld\n", n_total_load );
  // for ( int i=0; i<nproc; i++ ){
  //   if ( procID == i ) std::cout << "  [pId:"  << procID << "]  Loading Particles: " << n_local <<  std::endl;
  //   MPI_Barrier(world);
  // }
  MPI_Barrier(world);
  #endif

  
  dataset_buffer_px = (Real *) malloc(n_to_load*sizeof(Real));
  dataset_id = H5Dopen(file_id, "/pos_x", H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_px);
  status = H5Dclose(dataset_id);
  
  dataset_buffer_py = (Real *) malloc(n_to_load*sizeof(Real));
  dataset_id = H5Dopen(file_id, "/pos_y", H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_py);
  status = H5Dclose(dataset_id);
  
  dataset_buffer_pz = (Real *) malloc(n_to_load*sizeof(Real));
  dataset_id = H5Dopen(file_id, "/pos_z", H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_pz);
  status = H5Dclose(dataset_id);
  
  dataset_buffer_vx = (Real *) malloc(n_to_load*sizeof(Real));
  dataset_id = H5Dopen(file_id, "/vel_x", H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_vx);
  status = H5Dclose(dataset_id);
  
  dataset_buffer_vy = (Real *) malloc(n_to_load*sizeof(Real));
  dataset_id = H5Dopen(file_id, "/vel_y", H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_vy);
  status = H5Dclose(dataset_id);
  
  dataset_buffer_vz = (Real *) malloc(n_to_load*sizeof(Real));
  dataset_id = H5Dopen(file_id, "/vel_z", H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_vz);
  status = H5Dclose(dataset_id);
  
  #ifndef SINGLE_PARTICLE_MASS
  dataset_buffer_m = (Real *) malloc(n_to_load*sizeof(Real));
  dataset_id = H5Dopen(file_id, "/mass", H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_m);
  status = H5Dclose(dataset_id);
  #endif
  
  #ifdef PARTICLE_IDS
  part_int_t *dataset_buffer_IDs;
  dataset_buffer_IDs = (part_int_t *) malloc(n_to_load*sizeof(part_int_t));
  dataset_id = H5Dopen(file_id, "/particle_IDs", H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_IDs);
  status = H5Dclose(dataset_id);
  #endif
  
  Real px_min, px_max;
  Real py_min, py_max;
  Real pz_min, pz_max;
  Real vx_min, vx_max;
  Real vy_min, vy_max;
  Real vz_min, vz_max;
  px_min = 1e64;
  py_min = 1e64;
  pz_min = 1e64;
  px_max = -1e64;
  py_max = -1e64;
  pz_max = -1e64;
  vx_min = 1e64;
  vy_min = 1e64;
  vz_min = 1e64;
  vx_max = -1e64;
  vy_max = -1e64;
  vz_max = -1e64;
  
  
  
  
  
  Real pPos_x, pPos_y, pPos_z;
  Real pVel_x, pVel_y, pVel_z, pMass;
  part_int_t pID;
  bool in_local;

  #ifdef TILED_INITIAL_CONDITIONS
  
  Real Lx_local = G.xMax - G.xMin;
  Real Ly_local = G.yMax - G.yMin;
  Real Lz_local = G.zMax - G.zMin;
  
  Real tile_length = P->tile_length;
  // Rescale the particles position to the global domain
  chprintf(" Rescaling the Tiled Particles Positions... \n");
  chprintf("  Tile length:  %f   kpc/h \n", tile_length );
  chprintf("  N_Procs  Z: %d    Y: %d    X: %d  \n", nproc_z, nproc_y, nproc_x );
  
  bool tile_length_difference = false; 
  if ( fabs( Lx_local - tile_length ) / Lx_local > 1e-5  ) tile_length_difference = true;
  if ( fabs( Ly_local - tile_length ) / Ly_local > 1e-5  ) tile_length_difference = true;
  if ( fabs( Lz_local - tile_length ) / Lz_local > 1e-5  ) tile_length_difference = true;
  
  // if ( tile_length_difference ){
  //   std::cout << "  WARNING: Local Domain Length Different to Tile Length " << std::endl;
  //   printf("   Domain Length:  [ %f  %f  %f  ]\n", Lz_local, Ly_local, Lx_local );
  //   printf("   Tile Length:  %f \n", tile_length );
  // }
  
  #ifdef CUSTOM_DOMAIN_PFFT
  Real xblocal_cholla, yblocal_cholla, zblocal_cholla;
  xblocal_cholla = H.PFFT_Domain.xblocal_cholla;
  yblocal_cholla = H.PFFT_Domain.yblocal_cholla;
  zblocal_cholla = H.PFFT_Domain.zblocal_cholla;
  
  ptrdiff_t nx_local, ny_local, nz_local;
  nx_local = H.PFFT_Domain.nx_local;
  ny_local = H.PFFT_Domain.ny_local;
  nz_local = H.PFFT_Domain.nz_local;
  
  
  
  chprintf( "Domain:\n");
  for(int n=0; n<nproc; n++)
  {
    if(n==procID)
    {
      printf("procID %d [ %f %f ] [ %f %f ] [ %f %f ] \n", procID, G.xMin, G.xMax, G.yMin, G.yMax, G.zMin, G.zMax );
    } 
    fflush(stdout);
    MPI_Barrier(world);
    usleep(100);
  }
  
  chprintf( "xblocal:\n");
  for(int n=0; n<nproc; n++)
  {
    if(n==procID)
    {
      printf("procID %d [ %f  %f  %f  ] \n", procID, xblocal_cholla, yblocal_cholla, zblocal_cholla );
    } 
    fflush(stdout);
    MPI_Barrier(world);
    usleep(100);
  }
  
  
  // int k, j, i;
  
  for( k=0; k<nz_local; k++){
    for( j=0; j<ny_local; j++){
      for( i=0; i<nx_local; i++){
        
        in_local = true;
        
        pPos_x = G.xMin + i*G.dx + 0.5*G.dx;
        pPos_y = G.yMin + j*G.dy + 0.5*G.dy;
        pPos_z = G.zMin + k*G.dz + 0.5*G.dz;
        
        if ( pPos_x < G.xMin || pPos_x >= G.xMax ) in_local = false;
        if ( pPos_y < G.yMin || pPos_y >= G.yMax ) in_local = false;
        if ( pPos_z < G.zMin || pPos_z >= G.zMax ) in_local = false;
        if ( ! in_local  ) printf("Particle outside local domain\n" ); 
        
        pVel_x = 0;
        pVel_y = 0;
        pVel_z = 0;
        
        if  ( pPos_x > px_max ) px_max = pPos_x;
        if  ( pPos_y > py_max ) py_max = pPos_y;
        if  ( pPos_z > pz_max ) pz_max = pPos_z;
      
        if  ( pPos_x < px_min ) px_min = pPos_x;
        if  ( pPos_y < py_min ) py_min = pPos_y;
        if  ( pPos_z < pz_min ) pz_min = pPos_z;
      
        if  ( pVel_x > vx_max ) vx_max = pVel_x;
        if  ( pVel_y > vy_max ) vy_max = pVel_y;
        if  ( pVel_z > vz_max ) vz_max = pVel_z;
      
        if  ( pVel_x < vx_min ) vx_min = pVel_x;
        if  ( pVel_y < vy_min ) vy_min = pVel_y;
        if  ( pVel_z < vz_min ) vz_min = pVel_z;
      
        pos_x.push_back( pPos_x );
        pos_y.push_back( pPos_y );
        pos_z.push_back( pPos_z );
        vel_x.push_back( pVel_x );
        vel_y.push_back( pVel_y );
        vel_z.push_back( pVel_z );
        grav_x.push_back( 0.0 );
        grav_y.push_back( 0.0 );
        grav_z.push_back( 0.0 );

        n_local += 1;
        
        
      }
    }
  }
  
  
  #endif
  #endif
  
  
  
  #ifndef CUSTOM_DOMAIN_PFFT
  bool double_include;
  
  
  
  for( pIndx=0; pIndx<n_to_load; pIndx++ ){
    double_include = false;
    pPos_x = dataset_buffer_px[pIndx];
    pPos_y = dataset_buffer_py[pIndx];
    pPos_z = dataset_buffer_pz[pIndx];
    pVel_x = dataset_buffer_vx[pIndx];
    pVel_y = dataset_buffer_vy[pIndx];
    pVel_z = dataset_buffer_vz[pIndx];
    #ifndef SINGLE_PARTICLE_MASS
    pMass = dataset_buffer_m[pIndx];
    #endif
    #ifdef PARTICLE_IDS
    pID = dataset_buffer_IDs[pIndx];
    #endif
  
    #ifdef TILED_INITIAL_CONDITIONS
    // Rescale the particles position to the global domain
    // Move the particles to their position in Local Domain
    #ifdef CUSTOM_DOMAIN_PFFT
    pPos_x += xblocal_cholla;
    pPos_y += yblocal_cholla;
    pPos_z += zblocal_cholla;    
    #else    
    pPos_x += G.xMin;
    pPos_y += G.yMin;
    pPos_z += G.zMin;
    #endif
    #endif
  
    in_local = true;
    if ( pPos_x < G.domainMin_x || pPos_x > G.domainMax_x ){
      std::cout << " Particle outside global domain: "<< pPos_x << " [ " << G.domainMin_x << " , " << G.domainMax_x << " ]" << std::endl;
      
      // continue;
    }
    if ( pPos_y < G.domainMin_y || pPos_y > G.domainMax_y ){
      std::cout << " Particle outside global domain: "<< pPos_y << " [ " << G.domainMin_y << " , " << G.domainMax_y << " ]" << std::endl;
      // continue;
    }
    if ( pPos_z < G.domainMin_z || pPos_z > G.domainMax_z ){
      std::cout << " Particle outside global domain: "<< pPos_z << " [ " << G.domainMin_z << " , " << G.domainMax_z << " ]" << std::endl;
      // continue;
    }
    if ( pPos_x < G.xMin || pPos_x >= G.xMax ) in_local = false;
    if ( pPos_y < G.yMin || pPos_y >= G.yMax ) in_local = false;
    if ( pPos_z < G.zMin || pPos_z >= G.zMax ) in_local = false;
    // if ( ! in_local  ) {
      // #ifdef PARTICLE_IDS
      // std::cout << " Particle outside Loacal  domain    pID: " << pID << std::endl;
      // #else
      // std::cout << " Particle outside Loacal  domain " << std::endl;
      // #endif
      // std::cout << "  Domain X: " << G.xMin <<  "  " << G.xMax << std::endl;
      // std::cout << "  Domain Y: " << G.yMin <<  "  " << G.yMax << std::endl;
      // std::cout << "  Domain Z: " << G.zMin <<  "  " << G.zMax << std::endl;
      // std::cout << "  Particle X: " << pPos_x << std::endl;
      // std::cout << "  Particle Y: " << pPos_y << std::endl;
      // std::cout << "  Particle Z: " << pPos_z << std::endl;
    //   continue;
    // }
    // 
    // if ( pPos_x - Lx_local >= G.xMin || pPos_x + Lx_local < G.xMax ) double_include = true;
    // if ( pPos_y - Ly_local >= G.yMin || pPos_y + Ly_local < G.yMax ) double_include = true;
    // if ( pPos_z - Lz_local >= G.zMin || pPos_z + Lz_local < G.zMax ) double_include = true;
  
    if (in_local){
      if  ( pPos_x > px_max ) px_max = pPos_x;
      if  ( pPos_y > py_max ) py_max = pPos_y;
      if  ( pPos_z > pz_max ) pz_max = pPos_z;
    
      if  ( pPos_x < px_min ) px_min = pPos_x;
      if  ( pPos_y < py_min ) py_min = pPos_y;
      if  ( pPos_z < pz_min ) pz_min = pPos_z;
    
      if  ( pVel_x > vx_max ) vx_max = pVel_x;
      if  ( pVel_y > vy_max ) vy_max = pVel_y;
      if  ( pVel_z > vz_max ) vz_max = pVel_z;
    
      if  ( pVel_x < vx_min ) vx_min = pVel_x;
      if  ( pVel_y < vy_min ) vy_min = pVel_y;
      if  ( pVel_z < vz_min ) vz_min = pVel_z;
    
      pos_x.push_back( pPos_x );
      pos_y.push_back( pPos_y );
      pos_z.push_back( pPos_z );
      vel_x.push_back( pVel_x );
      vel_y.push_back( pVel_y );
      vel_z.push_back( pVel_z );
      grav_x.push_back( 0.0 );
      grav_y.push_back( 0.0 );
      grav_z.push_back( 0.0 );
      #ifndef SINGLE_PARTICLE_MASS
      mass.push_back( pMass );
      #endif
      #ifdef PARTICLE_IDS
      partIDs.push_back(pID);
      #endif
      n_local += 1;
    }
    
    if ( pPos_x - Lx_local >= G.xMin){
      pos_x.push_back( pPos_x - Lx_local );
      pos_y.push_back( pPos_y );
      pos_z.push_back( pPos_z );
      vel_x.push_back( pVel_x );
      vel_y.push_back( pVel_y );
      vel_z.push_back( pVel_z );
      grav_x.push_back( 0.0 );
      grav_y.push_back( 0.0 );
      grav_z.push_back( 0.0 );
      #ifndef SINGLE_PARTICLE_MASS
      mass.push_back( pMass );
      #endif
      #ifdef PARTICLE_IDS
      partIDs.push_back(pID);
      #endif
      n_local += 1;
    }
    
    if ( pPos_x + Lx_local < G.xMax){
      pos_x.push_back( pPos_x + Lx_local );
      pos_y.push_back( pPos_y );
      pos_z.push_back( pPos_z );
      vel_x.push_back( pVel_x );
      vel_y.push_back( pVel_y );
      vel_z.push_back( pVel_z );
      grav_x.push_back( 0.0 );
      grav_y.push_back( 0.0 );
      grav_z.push_back( 0.0 );
      #ifndef SINGLE_PARTICLE_MASS
      mass.push_back( pMass );
      #endif
      #ifdef PARTICLE_IDS
      partIDs.push_back(pID);
      #endif
      n_local += 1;
    }
    
    if ( pPos_y - Ly_local >= G.yMin){
      pos_x.push_back( pPos_x );
      pos_y.push_back( pPos_y - Ly_local );
      pos_z.push_back( pPos_z );
      vel_x.push_back( pVel_x );
      vel_y.push_back( pVel_y );
      vel_z.push_back( pVel_z );
      grav_x.push_back( 0.0 );
      grav_y.push_back( 0.0 );
      grav_z.push_back( 0.0 );
      #ifndef SINGLE_PARTICLE_MASS
      mass.push_back( pMass );
      #endif
      #ifdef PARTICLE_IDS
      partIDs.push_back(pID);
      #endif
      n_local += 1;
    }
    
    if ( pPos_y + Ly_local < G.yMax){
      pos_x.push_back( pPos_x  );
      pos_y.push_back( pPos_y + Ly_local );
      pos_z.push_back( pPos_z );
      vel_x.push_back( pVel_x );
      vel_y.push_back( pVel_y );
      vel_z.push_back( pVel_z );
      grav_x.push_back( 0.0 );
      grav_y.push_back( 0.0 );
      grav_z.push_back( 0.0 );
      #ifndef SINGLE_PARTICLE_MASS
      mass.push_back( pMass );
      #endif
      #ifdef PARTICLE_IDS
      partIDs.push_back(pID);
      #endif
      n_local += 1;
    }
    
    if ( pPos_z - Lz_local >= G.zMin){
      pos_x.push_back( pPos_x );
      pos_y.push_back( pPos_y );
      pos_z.push_back( pPos_z - Lz_local );
      vel_x.push_back( pVel_x );
      vel_y.push_back( pVel_y );
      vel_z.push_back( pVel_z );
      grav_x.push_back( 0.0 );
      grav_y.push_back( 0.0 );
      grav_z.push_back( 0.0 );
      #ifndef SINGLE_PARTICLE_MASS
      mass.push_back( pMass );
      #endif
      #ifdef PARTICLE_IDS
      partIDs.push_back(pID);
      #endif
      n_local += 1;
    }
    
    if ( pPos_z + Lz_local < G.zMax){
      pos_x.push_back( pPos_x  );
      pos_y.push_back( pPos_y  );
      pos_z.push_back( pPos_z + Lz_local);
      vel_x.push_back( pVel_x );
      vel_y.push_back( pVel_y );
      vel_z.push_back( pVel_z );
      grav_x.push_back( 0.0 );
      grav_y.push_back( 0.0 );
      grav_z.push_back( 0.0 );
      #ifndef SINGLE_PARTICLE_MASS
      mass.push_back( pMass );
      #endif
      #ifdef PARTICLE_IDS
      partIDs.push_back(pID);
      #endif
      n_local += 1;
    }
    
  }
  
  #endif //CUSTOM_DOMAIN_PFFT
  
  
  #ifndef MPI_CHOLLA
  chprintf( " Loaded  %ld  particles\n", n_local );
  #else
  // for ( int i=0; i<nproc; i++ ){
  //   if ( procID == i ) std::cout << "  [pId:"  << procID << "]  N Particles Loaded: " << n_local <<  std::endl;
  //   MPI_Barrier(world);
  // }
  MPI_Barrier(world);
  part_int_t n_total_loaded;
  n_total_loaded = ReducePartIntSum( n_local );
  n_total_initial = n_total_loaded;
  chprintf( " Total Particles Loaded: %ld\n", n_total_loaded );
  #endif
  
  #ifdef MPI_CHOLLA
  Real px_max_g = ReduceRealMax( px_max );
  Real py_max_g = ReduceRealMax( py_max );
  Real pz_max_g = ReduceRealMax( pz_max );
  Real vx_max_g = ReduceRealMax( vx_max );
  Real vy_max_g = ReduceRealMax( vy_max );
  Real vz_max_g = ReduceRealMax( vz_max );
  
  Real px_min_g = ReduceRealMin( px_min );
  Real py_min_g = ReduceRealMin( py_min );
  Real pz_min_g = ReduceRealMin( pz_min );
  Real vx_min_g = ReduceRealMin( vx_min );
  Real vy_min_g = ReduceRealMin( vy_min );
  Real vz_min_g = ReduceRealMin( vz_min );
  
  #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY)
  chprintf( "  Pos X   Min: %f   Max: %f   [ kpc/h ]\n", px_min_g, px_max_g);
  chprintf( "  Pos Y   Min: %f   Max: %f   [ kpc/h ]\n", py_min_g, py_max_g);
  chprintf( "  Pos Z   Min: %f   Max: %f   [ kpc/h ]\n", pz_min_g, pz_max_g);
  chprintf( "  Vel X   Min: %f   Max: %f   [ km/s ]\n", vx_min_g, vx_max_g);
  chprintf( "  Vel Y   Min: %f   Max: %f   [ km/s ]\n", vy_min_g, vy_max_g);
  chprintf( "  Vel Z   Min: %f   Max: %f   [ km/s ]\n", vz_min_g, vz_max_g);
  #endif//PRINT_INITIAL_STATS
  #endif//MPI_CHOLLA
  // 
  
  free(dataset_buffer_px);
  free(dataset_buffer_py);
  free(dataset_buffer_pz);
  free(dataset_buffer_vx);
  free(dataset_buffer_vy);
  free(dataset_buffer_vz);
  // 
  #ifndef SINGLE_PARTICLE_MASS
  free(dataset_buffer_m);
  #endif
  #ifdef PARTICLE_IDS
  free(dataset_buffer_IDs);
  #endif
} 




/*! \fn void Write_Header_HDF5(hid_t file_id)
 *  \brief Write the relevant header info to the HDF5 file. */
void Grid3D::Write_Particles_Header_HDF5( hid_t file_id){
  hid_t     attribute_id, dataspace_id;
  herr_t    status;
  hsize_t   attr_dims;
  int       int_data[3];
  Real      Real_data[3];

  // Single attributes first
  attr_dims = 1;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);
  // Create a group attribute
  attribute_id = H5Acreate(file_id, "t_particles", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  // Write the attribute data
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.t);
  // Close the attribute
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "dt_particles", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.dt);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "n_particles_local", H5T_STD_I64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_ULONG, &Particles.n_local);
  status = H5Aclose(attribute_id);
  // attribute_id = H5Acreate(file_id, "n_particles_total", H5T_STD_I64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  // status = H5Awrite(attribute_id, H5T_NATIVE_ULONG, &Particles.n_total);
  // status = H5Aclose(attribute_id);
  #ifdef COSMOLOGY
  attribute_id = H5Acreate(file_id, "current_a", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.current_a);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "current_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.current_z);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "h0", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.cosmo_h);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "omega_m,", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.Omega_M);
  status = H5Aclose(attribute_id);
  attribute_id = H5Acreate(file_id, "omega_l", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Cosmo.Omega_L);
  status = H5Aclose(attribute_id);
  
  
  #endif

  #ifdef SINGLE_PARTICLE_MASS
  attribute_id = H5Acreate(file_id, "particle_mass", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Particles.particle_mass);
  status = H5Aclose(attribute_id);
  #endif

  status = H5Sclose(dataspace_id);
}


void Grid3D::Write_Particles_Data_HDF5( hid_t file_id){
  part_int_t i, j, k, id, buf_id;
  hid_t     dataset_id, dataspace_id;
  Real      *dataset_buffer;
  part_int_t  *dataset_buffer_IDs;
  herr_t    status;
  part_int_t n_local = Particles.n_local;
  // int       nx_dset = H.nx_real;
  hsize_t   dims[1];
  dataset_buffer = (Real *) malloc(n_local*sizeof(Real));
  
  // Count Total Particles
  part_int_t N_paricles_total;
  #ifdef MPI_CHOLLA
  N_paricles_total = ReducePartIntSum( Particles.n_local );
  #else
  N_paricles_total = Particles.n_local;
  #endif
  
  chprintf( " Total Particles: %ld\n", N_paricles_total );
  
  if ( N_paricles_total != Particles.n_total_initial ) chprintf( " WARNING: Lost Particles: %d \n", Particles.n_total_initial - N_paricles_total );


  // Create the data space for the datasets
  dims[0] = n_local;
  dataspace_id = H5Screate_simple(1, dims, NULL);

  // Copy the pos_x vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.pos_x[i];
  dataset_id = H5Dcreate(file_id, "/pos_x", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the pos_y vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.pos_y[i];
  dataset_id = H5Dcreate(file_id, "/pos_y", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the pos_z vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.pos_z[i];
  dataset_id = H5Dcreate(file_id, "/pos_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the vel_x vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.vel_x[i];
  dataset_id = H5Dcreate(file_id, "/vel_x", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the vel_y vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.vel_y[i];
  dataset_id = H5Dcreate(file_id, "/vel_y", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  // Copy the vel_z vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.vel_z[i];
  dataset_id = H5Dcreate(file_id, "/vel_z", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  #ifndef SINGLE_PARTICLE_MASS
  // Copy the mass vector to the memory buffer
  for ( i=0; i<n_local; i++) dataset_buffer[i] = Particles.mass[i];
  dataset_id = H5Dcreate(file_id, "/mass", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);
  #endif

  #ifdef PARTICLE_IDS
  dataset_buffer_IDs = (part_int_t *) malloc(n_local*sizeof(part_int_t));
  for ( i=0; i<n_local; i++) dataset_buffer_IDs[i] = Particles.partIDs[i];
  dataset_id = H5Dcreate(file_id, "/particle_IDs", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer_IDs);
  status = H5Dclose(dataset_id);
  free(dataset_buffer_IDs);
  #endif

  // 3D case
  int       nx_dset = Particles.G.nx_local;
  int       ny_dset = Particles.G.ny_local;
  int       nz_dset = Particles.G.nz_local;
  hsize_t   dims3d[3];
  dataset_buffer = (Real *) malloc(Particles.G.nz_local*Particles.G.ny_local*Particles.G.nx_local*sizeof(Real));

  // Create the data space for the datasets
  dims3d[0] = nx_dset;
  dims3d[1] = ny_dset;
  dims3d[2] = nz_dset;
  dataspace_id = H5Screate_simple(3, dims3d, NULL);

  // Copy the density array to the memory buffer
  int nGHST = Particles.G.n_ghost_particles_grid;
  for (k=0; k<Particles.G.nz_local; k++) {
    for (j=0; j<Particles.G.ny_local; j++) {
      for (i=0; i<Particles.G.nx_local; i++) {
        id = (i+nGHST) + (j+nGHST)*(Particles.G.nx_local+2*nGHST) + (k+nGHST)*(Particles.G.nx_local+2*nGHST)*(Particles.G.ny_local+2*nGHST);
        buf_id = k + j*Particles.G.nz_local + i*Particles.G.nz_local*Particles.G.ny_local;
        dataset_buffer[buf_id] = Particles.G.density[id];
      }
    }
  }

  dataset_id = H5Dcreate(file_id, "/density", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);

  #ifdef OUTPUT_POTENTIAL
  // Copy the potential array to the memory buffer
  for (k=0; k<Grav.nz_local; k++) {
    for (j=0; j<Grav.ny_local; j++) {
      for (i=0; i<Grav.nx_local; i++) {
        id = (i+N_GHOST_POTENTIAL) + (j+N_GHOST_POTENTIAL)*(Grav.nx_local+2*N_GHOST_POTENTIAL) + (k+N_GHOST_POTENTIAL)*(Grav.nx_local+2*N_GHOST_POTENTIAL)*(Grav.ny_local+2*N_GHOST_POTENTIAL);
        buf_id = k + j*Grav.nz_local + i*Grav.nz_local*Grav.ny_local;
        dataset_buffer[buf_id] = Grav.F.potential_h[id];

      }
    }
  }
  dataset_id = H5Dcreate(file_id, "/grav_potential", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status = H5Dclose(dataset_id);
  #endif //OUTPUT_POTENTIAL


  free(dataset_buffer);
}
#endif//HDF5



void Grid3D::OutputData_Particles( struct parameters P, int nfile)
{
  FILE *out;
  char filename[80];
  char timestep[20];

  // create the filename
  strcpy(filename, P.outdir);
  sprintf(timestep, "%d", nfile);
  strcat(filename,timestep);
  // a binary file is created for each process
  #if defined BINARY
  chprintf("\nERROR: Particles only support HDF5 outputs\n")
  return;
  // only one HDF5 file is created
  #elif defined HDF5
  strcat(filename,"_particles");
  strcat(filename,".h5");
  #ifdef MPI_CHOLLA
  sprintf(filename,"%s.%d",filename,procID);
  #endif
  #endif

  #if defined HDF5
  hid_t   file_id;
  herr_t  status;

  // Create a new file collectively
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  Write_Header_HDF5(file_id);
  Write_Particles_Header_HDF5( file_id);
  Write_Particles_Data_HDF5( file_id);

  // Close the file
  status = H5Fclose(file_id);
  #endif
}



#endif