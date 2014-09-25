

/* Top-level kernel generation: will generate through multiple inclusion the
 * following flavors for all kernels:
 * - force-only output;
 * - force and energy output;
 * - force-only with pair list pruning;
 * - force and energy output with pair list pruning.
 */
/** Force only **/
//#include "nbnxn_ocl_kernels.clh"
/** Force & energy **/
#define CALC_ENERGIES
#include "nbnxn_ocl_kernels.clh"
#undef CALC_ENERGIES

/*** Pair-list pruning kernels ***/
/** Force only **/
#define PRUNE_NBL
//#include "nbnxn_ocl_kernels.clh"
/** Force & energy **/
#define CALC_ENERGIES
//#include "nbnxn_ocl_kernels.clh"
#undef CALC_ENERGIES
#undef PRUNE_NBL