/**
 * $Id: emdata.cpp,v 1.565 2015/05/30 05:46:22 stevel Exp $
 */

/*
 * Author: Steven Ludtke, 04/10/2003 (sludtke@bcm.edu)
 * Copyright (c) 2000-2006 Baylor College of Medicine
 *
 * This software is issued under a joint BSD/GNU license. You may use the
 * source code in this file under either license. However, note that the
 * complete EMAN2 and SPARX software packages have some GPL dependencies,
 * so you are responsible for compliance with the licenses of these packages
 * if you opt to use BSD licensing. The warranty disclaimer below holds
 * in either instance.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * */

#include "emdata.h"
#include "all_imageio.h"
#include "ctf.h"
#include "processor.h"
#include "aligner.h"
#include "cmp.h"
#include "emfft.h"
#include "projector.h"
#include "geometry.h"
#include <math.h>

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>

#include <iomanip>
#include <complex>

#include <algorithm> // fill
#include <cmath>

#ifdef WIN32
	#define M_PI 3.14159265358979323846f
#endif	//WIN32

#define EMDATA_EMAN2_DEBUG 0

#ifdef EMAN2_USING_CUDA
//#include <driver_functions.h>
#include "cuda/cuda_processor.h"
#include "cuda/cuda_emfft.h"
#endif // EMAN2_USING_CUDA

using namespace EMAN;
using namespace std;
using namespace boost;

int EMData::totalalloc=0;		// mainly used for debugging/memory leak purposes

EMData::EMData() :
#ifdef EMAN2_USING_CUDA
		cudarwdata(0), cudarodata(0), num_bytes(0), nextlistitem(0), prevlistitem(0), roneedsupdate(0), cudadirtybit(0),
#endif //EMAN2_USING_CUDA
#ifdef FFT_CACHING
	fftcache(0),
#endif //FFT_CACHING
		attr_dict(), rdata(0), supp(0), flags(0), changecount(0), nx(0), ny(0), nz(0), nxy(0), nxyz(0), xoff(0), yoff(0),
		zoff(0), all_translation(),	path(""), pathnum(0), rot_fp(0)

{
	ENTERFUNC;

	attr_dict["apix_x"] = 1.0f;
	attr_dict["apix_y"] = 1.0f;
	attr_dict["apix_z"] = 1.0f;

	attr_dict["is_complex"] = int(0);
	attr_dict["is_complex_x"] = int(0);
	attr_dict["is_complex_ri"] = int(1);

	attr_dict["datatype"] = (int)EMUtil::EM_FLOAT;

	EMData::totalalloc++;
#ifdef MEMDEBUG2
	printf("EMDATA+  %4d    %p\n",EMData::totalalloc,this);
#endif
	
	EXITFUNC;
}

EMData::EMData(const string& filename, int image_index) :
#ifdef EMAN2_USING_CUDA
		cudarwdata(0), cudarodata(0), num_bytes(0), nextlistitem(0), prevlistitem(0), roneedsupdate(0), cudadirtybit(0),
#endif //EMAN2_USING_CUDA
#ifdef FFT_CACHING
	fftcache(0),
#endif //FFT_CACHING
		attr_dict(), rdata(0), supp(0), flags(0), changecount(0), nx(0), ny(0), nz(0), nxy(0), nxyz(0), xoff(0), yoff(0), zoff(0),
		all_translation(),	path(filename), pathnum(image_index), rot_fp(0)
{
	ENTERFUNC;

	attr_dict["apix_x"] = 1.0f;
	attr_dict["apix_y"] = 1.0f;
	attr_dict["apix_z"] = 1.0f;

	attr_dict["is_complex"] = int(0);
	attr_dict["is_complex_x"] = int(0);
	attr_dict["is_complex_ri"] = int(1);

	attr_dict["datatype"] = (int)EMUtil::EM_FLOAT;

	this->read_image(filename, image_index);

	update();
	EMData::totalalloc++;
#ifdef MEMDEBUG2
	printf("EMDATA+  %4d    %p\n",EMData::totalalloc,this);
#endif

	EXITFUNC;
}

EMData::EMData(const EMData& that) :
#ifdef EMAN2_USING_CUDA
		cudarwdata(0), cudarodata(0), num_bytes(0), nextlistitem(0), prevlistitem(0), roneedsupdate(0), cudadirtybit(0),
#endif //EMAN2_USING_CUDA
#ifdef FFT_CACHING
	fftcache(0),
#endif //FFT_CACHING
		attr_dict(that.attr_dict), rdata(0), supp(0), flags(that.flags), changecount(that.changecount), nx(that.nx), ny(that.ny), nz(that.nz),
		nxy(that.nx*that.ny), nxyz((size_t)that.nx*that.ny*that.nz), xoff(that.xoff), yoff(that.yoff), zoff(that.zoff),all_translation(that.all_translation),	path(that.path),
		pathnum(that.pathnum), rot_fp(0)
{
	ENTERFUNC;
	
	float* data = that.rdata;
	size_t num_bytes = (size_t)nx*ny*nz*sizeof(float);
	if (data && num_bytes != 0)
	{
		rdata = (float*)EMUtil::em_malloc(num_bytes);
		EMUtil::em_memcpy(rdata, data, num_bytes);
	}
#ifdef EMAN2_USING_CUDA
	if (EMData::usecuda == 1 && num_bytes != 0 && that.cudarwdata != 0) {
		//cout << "That copy constructor" << endl;
		if(!rw_alloc()) throw UnexpectedBehaviorException("Bad alloc");
		cudaError_t error = cudaMemcpy(cudarwdata,that.cudarwdata,num_bytes,cudaMemcpyDeviceToDevice);
		if ( error != cudaSuccess ) throw UnexpectedBehaviorException("cudaMemcpy failed in EMData copy construction with error: " + string(cudaGetErrorString(error)));
	}
#endif //EMAN2_USING_CUDA

	if (that.rot_fp != 0) rot_fp = new EMData(*(that.rot_fp));

	EMData::totalalloc++;
#ifdef MEMDEBUG2
	printf("EMDATA+  %4d    %p\n",EMData::totalalloc,this);
#endif

	ENTERFUNC;
}

EMData& EMData::operator=(const EMData& that)
{
	ENTERFUNC;

	if ( this != &that )
	{
		free_memory(); // Free memory sets nx,ny and nz to 0

		// Only copy the rdata if it exists, we could be in a scenario where only the header has been read
		float* data = that.rdata;
		size_t num_bytes = that.nx*that.ny*that.nz*sizeof(float);
		if (data && num_bytes != 0)
		{
			nx = 1; // This prevents a memset in set_size
			set_size(that.nx,that.ny,that.nz);
			EMUtil::em_memcpy(rdata, data, num_bytes);
		}

		flags = that.flags;

		all_translation = that.all_translation;

		path = that.path;
		pathnum = that.pathnum;
		attr_dict = that.attr_dict;

		xoff = that.xoff;
		yoff = that.yoff;
		zoff = that.zoff;

#ifdef EMAN2_USING_CUDA
		if (EMData::usecuda == 1 && num_bytes != 0 && that.cudarwdata != 0) {
			if(cudarwdata){rw_free();}
			if(!rw_alloc()) throw UnexpectedBehaviorException("Bad alloc");;
			cudaError_t error = cudaMemcpy(cudarwdata,that.cudarwdata,num_bytes,cudaMemcpyDeviceToDevice);
			if ( error != cudaSuccess ) throw UnexpectedBehaviorException("cudaMemcpy failed in EMData copy construction with error: " + string(cudaGetErrorString(error)));
		}
#endif //EMAN2_USING_CUDA

		changecount = that.changecount;

		if (that.rot_fp != 0) rot_fp = new EMData(*(that.rot_fp));
		else rot_fp = 0;
	}
	EXITFUNC;
	return *this;
}

EMData::EMData(int nx, int ny, int nz, bool is_real) :
#ifdef EMAN2_USING_CUDA
		cudarwdata(0), cudarodata(0), num_bytes(0), nextlistitem(0), prevlistitem(0), roneedsupdate(0), cudadirtybit(0),
#endif //EMAN2_USING_CUDA
#ifdef FFT_CACHING
	fftcache(0),
#endif //FFT_CACHING
		attr_dict(), rdata(0), supp(0), flags(0), changecount(0), nx(0), ny(0), nz(0), nxy(0), nxyz(0), xoff(0), yoff(0), zoff(0),
		all_translation(),	path(""), pathnum(0), rot_fp(0)
{
	ENTERFUNC;

	// used to replace cube 'pixel'
	attr_dict["apix_x"] = 1.0f;
	attr_dict["apix_y"] = 1.0f;
	attr_dict["apix_z"] = 1.0f;

	if(is_real) {	// create a real image [nx, ny, nz]
		attr_dict["is_complex"] = int(0);
		attr_dict["is_complex_x"] = int(0);
		attr_dict["is_complex_ri"] = int(1);
		set_size(nx, ny, nz);
	}
	else {	//create a complex image which real dimension is [ny, ny, nz]
		int new_nx = nx + 2 - nx%2;
		set_size(new_nx, ny, nz);

		attr_dict["is_complex"] = int(1);

		if(ny==1 && nz ==1)	{
			attr_dict["is_complex_x"] = int(1);
		}
		else {
			attr_dict["is_complex_x"] = int(0);
		}

		attr_dict["is_complex_ri"] = int(1);
		attr_dict["is_fftpad"] = int(1);

		if(nx%2 == 1) {
			attr_dict["is_fftodd"] = 1;
		}
	}

	this->to_zero();
	update();
	EMData::totalalloc++;
#ifdef MEMDEBUG2
	printf("EMDATA+  %4d    %p\n",EMData::totalalloc,this);
#endif

	EXITFUNC;
}


EMData::EMData(float* data, const int x, const int y, const int z, const Dict& attr_dict) :
#ifdef EMAN2_USING_CUDA
		cudarwdata(0), cudarodata(0), num_bytes(0), nextlistitem(0), prevlistitem(0), roneedsupdate(0), cudadirtybit(0),
#endif //EMAN2_USING_CUDA
#ifdef FFT_CACHING
	fftcache(0),
#endif //FFT_CACHING
		attr_dict(attr_dict), rdata(data), supp(0), flags(0), changecount(0), nx(x), ny(y), nz(z), nxy(x*y), nxyz((size_t)x*y*z), xoff(0),
		yoff(0), zoff(0), all_translation(), path(""), pathnum(0), rot_fp(0)
{
	ENTERFUNC;
	// used to replace cube 'pixel'
	attr_dict["apix_x"] = 1.0f;
	attr_dict["apix_y"] = 1.0f;
	attr_dict["apix_z"] = 1.0f;

	EMData::totalalloc++;
#ifdef MEMDEBUG2
	printf("EMDATA+  %4d    %p\n",EMData::totalalloc,this);
#endif

	update();
	EXITFUNC;
}

#ifdef EMAN2_USING_CUDA

EMData::EMData(float* data, float* cudadata, const int x, const int y, const int z, const Dict& attr_dict) :
		cudarwdata(cudadata), cudarodata(0), num_bytes(x*y*z*sizeof(float)), nextlistitem(0), prevlistitem(0), roneedsupdate(0), cudadirtybit(1),
#ifdef FFT_CACHING
	fftcache(0),
#endif //FFT_CACHING
		attr_dict(attr_dict), rdata(data), supp(0), flags(0), changecount(0), nx(x), ny(y), nz(z), nxy(x*y), nxyz((size_t)x*y*z), xoff(0),
		yoff(0), zoff(0), all_translation(), path(""), pathnum(0), rot_fp(0)
{
	ENTERFUNC;

	// used to replace cube 'pixel'
	attr_dict["apix_x"] = 1.0f;
	attr_dict["apix_y"] = 1.0f;
	attr_dict["apix_z"] = 1.0f;

	EMData::totalalloc++;
#ifdef MEMDEBUG2
	printf("EMDATA+  %4d    %p\n",EMData::totalalloc,this);
#endif
	update();
	EXITFUNC;
}

#endif //EMAN2_USING_CUDA

//debug
using std::cout;
using std::endl;
EMData::~EMData()
{
	ENTERFUNC;
#ifdef FFT_CACHING
	if (fftcache!=0) { delete fftcache; fftcache=0;}
#endif //FFT_CACHING
	free_memory();

#ifdef EMAN2_USING_CUDA
	if(cudarwdata){rw_free();}
	if(cudarodata){ro_free();}
#endif // EMAN2_USING_CUDA
	EMData::totalalloc--;
#ifdef MEMDEBUG2
	printf("EMDATA-  %4d    %p\n",EMData::totalalloc,this);
#endif
	EXITFUNC;
}

void EMData::clip_inplace(const Region & area,const float& fill_value)
{
	// Added by d.woolford
	ENTERFUNC;

//	printf("cip %d %d %d %d %d %d %f %d %d %d\n",area.origin[0],area.origin[1],area.origin[2],area.size[0],area.size[1],area.size[2],fill_value,nx,ny,nz);
	// Store the current dimension values
	int prev_nx = nx, prev_ny = ny, prev_nz = nz;
	size_t prev_size = (size_t)nx*ny*nz;

	// Get the zsize, ysize and xsize of the final area, these are the new dimension sizes of the pixel data
	int new_nz = ( area.size[2]==0 ? 1 : (int)area.size[2]);
	int new_ny = ( area.size[1]==0 ? 1 : (int)area.size[1]);
	int new_nx = (int)area.size[0];

	if ( new_nz < 0 || new_ny < 0 || new_nx < 0 )
	{
		// Negative image dimensions were never tested nor considered when creating this implementation
		throw ImageDimensionException("New image dimensions are negative - this is not supported in the clip_inplace operation");
	}

	size_t new_size = (size_t)new_nz*new_ny*new_nx;

	// Get the translation values, they are used to construct the ClipInplaceVariables object
	int x0 = (int) area.origin[0];
	int y0 = (int) area.origin[1];
	int z0 = (int) area.origin[2];

	// Get a object that calculates all the interesting variables associated with the clip inplace operation
	ClipInplaceVariables civ(prev_nx, prev_ny, prev_nz, new_nx, new_ny, new_nz, x0, y0, z0);

	get_data(); // Do this here to make sure rdata is up to date, applicable if GPU stuff is occuring
	// Double check to see if any memory shifting even has to occur
	if ( x0 > prev_nx || y0 > prev_ny || z0 > prev_nz || civ.x_iter == 0 || civ.y_iter == 0 || civ.z_iter == 0)
	{
		// In this case the volume has been shifted beyond the location of the pixel rdata and
		// the client should expect to see a volume with nothing in it.

		// Set size calls realloc,
		set_size(new_nx, new_ny, new_nz);

		// Set pixel memory to zero - the client should expect to see nothing
		EMUtil::em_memset(rdata, 0, (size_t)new_nx*new_ny*new_nz);

		return;
	}

	// Resize the volume before memory shifting occurs if the new volume is larger than the previous volume
	// All of the pixel rdata is guaranteed to be at the start of the new volume because realloc (called in set size)
	// guarantees this.
	if ( new_size > prev_size )
		set_size(new_nx, new_ny, new_nz);

	// Store the clipped row size.
	size_t clipped_row_size = (civ.x_iter) * sizeof(float);

	// Get the new sector sizes to save multiplication later.
	size_t new_sec_size = new_nx * new_ny;
	size_t prev_sec_size = prev_nx * prev_ny;

	// Determine the memory locations of the source and destination pixels - at the point nearest
	// to the beginning of the volume (rdata)
	size_t src_it_begin = civ.prv_z_bottom*prev_sec_size + civ.prv_y_front*prev_nx + civ.prv_x_left;
	size_t dst_it_begin = civ.new_z_bottom*new_sec_size + civ.new_y_front*new_nx + civ.new_x_left;

	// This loop is in the forward direction (starting at points nearest to the beginning of the volume)
	// it copies memory only when the destination pointer is less the source pointer - therefore
	// ensuring that no memory "copied to" is yet to be "copied from"
	for (int i = 0; i < civ.z_iter; ++i) {
		for (int j = 0; j < civ.y_iter; ++j) {

			// Determine the memory increments as dependent on i and j
			// This could be optimized so that not so many multiplications are occurring...
			size_t dst_inc = dst_it_begin + j*new_nx + i*new_sec_size;
			size_t src_inc = src_it_begin + j*prev_nx + i*prev_sec_size;
			float* local_dst = rdata + dst_inc;
			float* local_src = rdata + src_inc;

			if ( dst_inc >= src_inc )
			{
				// this is fine, it will happen now and then and it will be necessary to continue.
				// the tempatation is to break, but you can't do that (because the point where memory intersects
				// could be in this slice - and yes, this aspect could be optimized).
				continue;
			}

			// Asserts are compiled only in debug mode
			// This situation not encountered in testing thus far
			Assert( dst_inc < new_size && src_inc < prev_size && dst_inc >= 0 && src_inc >= 0 );

			// Finally copy the memory
			memmove(local_dst, local_src, clipped_row_size);
		}
	}

	// Determine the memory locations of the source and destination pixels - at the point nearest
	// to the end of the volume (rdata+new_size)
	size_t src_it_end = prev_size - civ.prv_z_top*prev_sec_size - civ.prv_y_back*prev_nx - prev_nx + civ.prv_x_left;
	size_t dst_it_end = new_size - civ.new_z_top*new_sec_size - civ.new_y_back*new_nx - new_nx + civ.new_x_left;

	// This loop is in the reverse direction (starting at points nearest to the end of the volume).
	// It copies memory only when the destination pointer is greater than  the source pointer therefore
	// ensuring that no memory "copied to" is yet to be "copied from"
	for (int i = 0; i < civ.z_iter; ++i) {
		for (int j = 0; j < civ.y_iter; ++j) {

			// Determine the memory increments as dependent on i and j
			size_t dst_inc = dst_it_end - j*new_nx - i*new_sec_size;
			size_t src_inc = src_it_end - j*prev_nx - i*prev_sec_size;
			float* local_dst = rdata + dst_inc;
			float* local_src = rdata + src_inc;

			if (dst_inc <= (src_inc + civ.x_iter ))
			{
				// Overlap
				if ( dst_inc > src_inc )
				{
					// Because the memcpy operation is the forward direction, and this "reverse
					// direction" loop is proceeding in a backwards direction, it is possible
					// that memory copied to is yet to be copied from (because memcpy goes forward).
					// In this scenario pixel memory "copied to" is yet to be "copied from"
					// i.e. there is overlap

					// memmove handles overlapping cases.
					// memmove could use a temporary buffer, or could go just go backwards
					// the specification doesn't say how the function behaves...
					// If memmove creates a temporary buffer is clip_inplace no longer inplace?
					memmove(local_dst, local_src, clipped_row_size);
				}
				continue;
			}

			// This situation not encountered in testing thus far
			Assert( dst_inc < new_size && src_inc < prev_size && dst_inc >= 0 && src_inc >= 0 );

			// Perform the memory copy
			EMUtil::em_memcpy(local_dst, local_src, clipped_row_size);
		}
	}

	// Resize the volume after memory shifting occurs if the new volume is smaller than the previous volume
	// set_size calls realloc, guaranteeing that the pixel rdata is in the right location.
	if ( new_size < prev_size )
		set_size(new_nx, new_ny, new_nz);

	// Now set all the edges to zero

	// Set the extra bottom z slices to the fill_value
	if (  z0 < 0 )
	{
		//EMUtil::em_memset(rdata, 0, (-z0)*new_sec_size*sizeof(float));
		size_t inc = (-z0)*new_sec_size;
		std::fill(rdata,rdata+inc,fill_value);
	}

	// Set the extra top z slices to the fill_value
	if (  civ.new_z_top > 0 )
	{
		float* begin_pointer = rdata + (new_nz-civ.new_z_top)*new_sec_size;
		//EMUtil::em_memset(begin_pointer, 0, (civ.new_z_top)*new_sec_size*sizeof(float));
		float* end_pointer = begin_pointer+(civ.new_z_top)*new_sec_size;
		std::fill(begin_pointer,end_pointer,fill_value);
	}

	// Next deal with x and y edges by iterating through each slice
	for ( int i = civ.new_z_bottom; i < civ.new_z_bottom + civ.z_iter; ++i )
	{
		// Set the extra front y components to the fill_value
		if ( y0 < 0 )
		{
			float* begin_pointer = rdata + i*new_sec_size;
			//EMUtil::em_memset(begin_pointer, 0, (-y0)*new_nx*sizeof(float));
			float* end_pointer = begin_pointer+(-y0)*new_nx;
			std::fill(begin_pointer,end_pointer,fill_value);
		}

		// Set the extra back y components to the fill_value
		if ( civ.new_y_back > 0 )
		{
			float* begin_pointer = rdata + i*new_sec_size + (new_ny-civ.new_y_back)*new_nx;
			//EMUtil::em_memset(begin_pointer, 0, (civ.new_y_back)*new_nx*sizeof(float));
			float* end_pointer = begin_pointer+(civ.new_y_back)*new_nx;
			std::fill(begin_pointer,end_pointer,fill_value);
		}

		// Iterate through the y to set each correct x component to the fill_value
		for (int j = civ.new_y_front; j <civ.new_y_front + civ.y_iter; ++j)
		{
			// Set the extra left x components to the fill_value
			if ( x0 < 0 )
			{
				float* begin_pointer = rdata + i*new_sec_size + j*new_nx;
				//EMUtil::em_memset(begin_pointer, 0, (-x0)*sizeof(float));
				float* end_pointer = begin_pointer+(-x0);
				std::fill(begin_pointer,end_pointer,fill_value);
			}

			// Set the extra right x components to the fill_value
			if ( civ.new_x_right > 0 )
			{
				float* begin_pointer = rdata + i*new_sec_size + j*new_nx + (new_nx - civ.new_x_right);
				//EMUtil::em_memset(begin_pointer, 0, (civ.new_x_right)*sizeof(float));
				float* end_pointer = begin_pointer+(civ.new_x_right);
				std::fill(begin_pointer,end_pointer,fill_value);
			}

		}
	}

// These couts may be useful
// 	cout << "start starts " << civ.prv_x_left << " " << civ.prv_y_front << " " << civ.prv_z_bottom << endl;
// 	cout << "start ends " << civ.prv_x_right << " " << civ.prv_y_back << " " << civ.prv_z_top << endl;
// 	cout << "dst starts " << civ.new_x_left << " " << civ.new_y_front << " " << civ.new_z_bottom << endl;
// 	cout << "dst ends " << civ.new_x_right << " " << civ.new_y_back << " " << civ.new_z_top << endl;
// 	cout << "total iter z - " << civ.z_iter << " y - " << civ.y_iter << " x - " << civ.x_iter << endl;
// 	cout << "=====" << endl;
// 	cout << "dst_end is " << dst_it_end << " src end is " << src_it_end << endl;
// 	cout << "dst_begin is " << dst_it_begin << " src begin is " << src_it_begin << endl;

	// Update appropriate attributes (Copied and pasted from get_clip)
	if( attr_dict.has_key("origin_x") && attr_dict.has_key("origin_y") &&
	attr_dict.has_key("origin_z") )
	{
		float xorigin = attr_dict["origin_x"];
		float yorigin = attr_dict["origin_y"];
		float zorigin = attr_dict["origin_z"];

		float apix_x = attr_dict["apix_x"];
		float apix_y = attr_dict["apix_y"];
		float apix_z = attr_dict["apix_z"];

		set_xyz_origin(xorigin + apix_x * area.origin[0],
			yorigin + apix_y * area.origin[1],
			zorigin + apix_z * area.origin[2]);
	}

	// Set the update flag because the size of the image has changed and stats should probably be recalculated if requested.
	update();

	EXITFUNC;
}

EMData *EMData::get_clip(const Region & area, const float fill) const
{
	ENTERFUNC;
	if (get_ndim() != area.get_ndim()) {
		LOGERR("cannot get %dD clip out of %dD image", area.get_ndim(),get_ndim());
		return 0;
	}

	EMData *result = new EMData();

	// Ensure that all of the metadata of this is stored in the new object
	// Originally added to ensure that euler angles were retained when preprocessing (zero padding) images
	// prior to insertion into the 3D for volume in the reconstruction phase (see reconstructor.cpp/h).
	result->attr_dict = this->attr_dict;
	int zsize = (int)area.size[2];
	if (zsize == 0 && nz <= 1) {
		zsize = 1;
	}
	int ysize = (ny<=1 && (int)area.size[1]==0 ? 1 : (int)area.size[1]);

	if ( (int)area.size[0] < 0 || ysize < 0 || zsize < 0 )
	{
		// Negative image dimensions not supported - added retrospectively by d.woolford (who didn't write get_clip but wrote clip_inplace)
		throw ImageDimensionException("New image dimensions are negative - this is not supported in the the get_clip operation");
	}

//#ifdef EMAN2_USING_CUDA
	// Strategy is always to prefer using the GPU if possible
//	bool use_gpu = false;
//	if ( gpu_operation_preferred() ) {
//		result->set_size_cuda((int)area.size[0], ysize, zsize);
		//CudaDataLock lock(this); // Just so we never have to recopy this data to and from the GPU
//		result->get_cuda_data(); // Force the allocation - set_size_cuda is lazy
		// Setting the value is necessary seeing as cuda data is not automatically zeroed
//		result->to_value(fill); // This will automatically use the GPU.
//		use_gpu = true;
//	} else { // cpu == True
//		result->set_size((int)area.size[0], ysize, zsize);
//		if (fill != 0.0) { result->to_value(fill); };
//	}
//#else
	result->set_size((int)area.size[0], ysize, zsize);
	if (fill != 0.0) { result->to_value(fill); };
//#endif //EMAN2_USING_CUDA

	int x0 = (int) area.origin[0];
	x0 = x0 < 0 ? 0 : x0;

	int y0 = (int) area.origin[1];
	y0 = y0 < 0 ? 0 : y0;

	int z0 = (int) area.origin[2];
	z0 = z0 < 0 ? 0 : z0;

	int x1 = (int) (area.origin[0] + area.size[0]);
	x1 = x1 > nx ? nx : x1;

	int y1 = (int) (area.origin[1] + area.size[1]);
	y1 = y1 > ny ? ny : y1;

	int z1 = (int) (area.origin[2] + area.size[2]);
	z1 = z1 > nz ? nz : z1;
	if (z1 <= 0) {
		z1 = 1;
	}

	result->insert_clip(this,-((IntPoint)area.origin));

	if( attr_dict.has_key("apix_x") && attr_dict.has_key("apix_y") &&
		attr_dict.has_key("apix_z") )
	{
		if( attr_dict.has_key("origin_x") && attr_dict.has_key("origin_y") &&
		    attr_dict.has_key("origin_z") )
		{
			float xorigin = attr_dict["origin_x"];
			float yorigin = attr_dict["origin_y"];
			float zorigin = attr_dict["origin_z"];

			float apix_x = attr_dict["apix_x"];
			float apix_y = attr_dict["apix_y"];
			float apix_z = attr_dict["apix_z"];

			result->set_xyz_origin(xorigin + apix_x * area.origin[0],
							   	   yorigin + apix_y * area.origin[1],
							       zorigin + apix_z * area.origin[2]);
		}
	}

//#ifdef EMAN2_USING_CUDA
//	if (use_gpu) result->gpu_update();
//	else result->update();
//#else
	result->update();
//#endif // EMAN2_USING_CUDA


	result->set_path(path);
	result->set_pathnum(pathnum);

	EXITFUNC;
	return result;
}


EMData *EMData::get_top_half() const
{
	ENTERFUNC;

	if (get_ndim() != 3) {
		throw ImageDimensionException("3D only");
	}

	EMData *half = new EMData();
	half->attr_dict = attr_dict;
	half->set_size(nx, ny, nz / 2);

	float *half_data = half->get_data();
	EMUtil::em_memcpy(half_data, &(get_data()[(size_t)nz / 2 * (size_t)nx * (size_t)ny]), sizeof(float) * (size_t)nx * (size_t)ny * (size_t)nz / 2lu);

	float apix_z = attr_dict["apix_z"];
	float origin_z = attr_dict["origin_z"];
	origin_z += apix_z * nz / 2;
	half->attr_dict["origin_z"] = origin_z;
	half->update();

	EXITFUNC;
	return half;
}


EMData *EMData::get_rotated_clip(const Transform &xform,
								 const IntSize &size, float)
{
	EMData *result = new EMData();
	result->set_size(size[0],size[1],size[2]);

	if (nz==1) {
		for (int y=-size[1]/2; y<(size[1]+1)/2; y++) {
			for (int x=-size[0]/2; x<(size[0]+1)/2; x++) {
				Vec3f xv=xform.transform(Vec3f((float)x,(float)y,0.0f));
				float v = 0;

				if (xv[0]<0||xv[1]<0||xv[0]>nx-2||xv[1]>ny-2) v=0.;
				else v=sget_value_at_interp(xv[0],xv[1]);
				result->set_value_at(x+size[0]/2,y+size[1]/2,v);
			}
		}
	}
	else {
		for (int z=-size[2]/2; z<(size[2]+1)/2; z++) {
			for (int y=-size[1]/2; y<(size[1]+1)/2; y++) {
				for (int x=-size[0]/2; x<(size[0]+1)/2; x++) {
					Vec3f xv=xform.transform(Vec3f((float)x,(float)y,(float)z));
					float v = 0;

					if (xv[0]<0||xv[1]<0||xv[2]<0||xv[0]>nx-2||xv[1]>ny-2||xv[2]>nz-2) v=0.;
					else v=sget_value_at_interp(xv[0],xv[1],xv[2]);
					result->set_value_at(x+size[0]/2,y+size[1]/2,z+size[2]/2,v);
				}
			}
		}
	}
	result->update();

	return result;
}


EMData* EMData::window_center(int l) {
	ENTERFUNC;
	// sanity checks
	int n = nx;
	if (is_complex()) {
		LOGERR("Need real-space data for window_center()");
		throw ImageFormatException(
			"Complex input image; real-space expected.");
	}
	if (is_fftpadded()) {
		// image has been fft-padded, compute the real-space size
		n -= (2 - int(is_fftodd()));
	}
	int corner = n/2 - l/2;
	int ndim = get_ndim();
	EMData* ret;
	switch (ndim) {
		case 3:
			if ((n != ny) || (n != nz)) {
				LOGERR("Need the real-space image to be cubic.");
				throw ImageFormatException(
						"Need cubic real-space image.");
			}
			ret = get_clip(Region(corner, corner, corner, l, l, l));
			break;
		case 2:
			if (n != ny) {
				LOGERR("Need the real-space image to be square.");
				throw ImageFormatException(
						"Need square real-space image.");
			}
			//cout << "Using corner " << corner << endl;
			ret = get_clip(Region(corner, corner, l, l));
			break;
		case 1:
			ret = get_clip(Region(corner, l));
			break;
		default:
			throw ImageDimensionException(
					"window_center only supports 1-d, 2-d, and 3-d images");
	}
	return ret;
	EXITFUNC;
}


float *EMData::setup4slice(bool redo)
{
	ENTERFUNC;

	if (!is_complex()) {
		throw ImageFormatException("complex image only");
	}

	if (get_ndim() != 3) {
		throw ImageDimensionException("3D only");
	}

	if (supp) {
		if (redo) {
			EMUtil::em_free(supp);
			supp = 0;
		}
		else {
			EXITFUNC;
			return supp;
		}
	}

	const int SUPP_ROW_SIZE = 8;
	const int SUPP_ROW_OFFSET = 4;
	const int supp_size = SUPP_ROW_SIZE + SUPP_ROW_OFFSET;

	supp = (float *) EMUtil::em_calloc(supp_size * ny * nz, sizeof(float));
	int nxy = nx * ny;
	int supp_xy = supp_size * ny;
	float * data = get_data();

	for (int z = 0; z < nz; z++) {
		size_t cur_z1 = z * nxy;
		size_t cur_z2 = z * supp_xy;

		for (int y = 0; y < ny; y++) {
			size_t cur_y1 = y * nx;
			size_t cur_y2 = y * supp_size;

			for (int x = 0; x < SUPP_ROW_SIZE; x++) {
				size_t k = (x + SUPP_ROW_OFFSET) + cur_y2 + cur_z2;
				supp[k] = data[x + cur_y1 + cur_z1];
			}
		}
	}

	for (int z = 1, zz = nz - 1; z < nz; z++, zz--) {
		size_t cur_z1 = zz * nxy;
		size_t cur_z2 = z * supp_xy;

		for (int y = 1, yy = ny - 1; y < ny; y++, yy--) {
			supp[y * 12 + cur_z2] = data[4 + yy * nx + cur_z1];
			supp[1 + y * 12 + cur_z2] = -data[5 + yy * nx + cur_z1];
			supp[2 + y * 12 + cur_z2] = data[2 + yy * nx + cur_z1];
			supp[3 + y * 12 + cur_z2] = -data[3 + yy * nx + cur_z1];
		}
	}

	EXITFUNC;
	return supp;
}


void EMData::scale(float s)
{
	ENTERFUNC;
	Transform t;
	t.set_scale(s);
	transform(t);
	EXITFUNC;
}


void EMData::translate(int dx, int dy, int dz)
{
	ENTERFUNC;
	translate(Vec3i(dx, dy, dz));
	EXITFUNC;
}


void EMData::translate(float dx, float dy, float dz)
{
	ENTERFUNC;
 	int dx_ = Util::round(dx);
 	int dy_ = Util::round(dy);
 	int dz_ = Util::round(dz);
 	if( ( (dx-dx_) == 0 ) && ( (dy-dy_) == 0 ) && ( (dz-dz_) == 0 )) {
 		translate(dx_, dy_, dz_);
 	}
 	else {
		translate(Vec3f(dx, dy, dz));
 	}
	EXITFUNC;
}


void EMData::translate(const Vec3i &translation)
{
	ENTERFUNC;

	//if traslation is 0, do nothing
	if( translation[0] == 0 && translation[1] == 0 && translation[2] == 0) {
		EXITFUNC;
		return;
	}

	Dict params("trans",static_cast< vector<int> >(translation));
	process_inplace("xform.translate.int",params);

	// update() - clip_inplace does the update
	all_translation += translation;

	EXITFUNC;
}


void EMData::translate(const Vec3f &translation)
{
	ENTERFUNC;

	if( translation[0] == 0.0f && translation[1] == 0.0f && translation[2] == 0.0f ) {
		EXITFUNC;
		return;
	}

	Transform* t = new Transform();
	t->set_trans(translation);
	process_inplace("xform",Dict("transform",t));
	delete t;

	all_translation += translation;
	EXITFUNC;
}


void EMData::rotate(float az, float alt, float phi)
{
	Dict d("type","eman");
	d["az"] = az;
	d["alt"] = alt;
	d["phi"] = phi;
	Transform t(d);
	transform(t);
}



void EMData::rotate(const Transform & t)
{
	cout << "Deprecation warning in EMData::rotate. Please consider using EMData::transform() instead " << endl;
	transform(t);
}

float EMData::max_3D_pixel_error(const Transform &t1, const Transform & t2, float r) {

	Transform t;
	int r0 = (int)r;
	float ddmax = 0.0f;

	t = t2*t1.inverse();
	for (int i=0; i<int(2*M_PI*r0+0.5); i++) {
		float ang = (float)i/r;
		Vec3f v = Vec3f(r0*cos(ang), r0*sin(ang), 0.0f);
		Vec3f d = t*v-v;
#ifdef _WIN32
		ddmax = _cpp_max(ddmax,d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
#else
		ddmax = std::max(ddmax,d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
#endif	//_WIN32
	}
	return std::sqrt(ddmax);
}

void EMData::rotate_translate(float az, float alt, float phi, float dx, float dy, float dz)
{
	cout << "Deprecation warning in EMData::rotate_translate. Please consider using EMData::transform() instead " << endl;
//	Transform3D t( az, alt, phi,Vec3f(dx, dy, dz));
	Transform t;
	t.set_rotation(Dict("type", "eman", "az", az, "alt", alt, "phi", phi));
	t.set_trans(dx, dy, dz);
	rotate_translate(t);
}


void EMData::rotate_translate(float az, float alt, float phi, float dx, float dy,
							  float dz, float pdx, float pdy, float pdz)
{
	cout << "Deprecation warning in EMData::rotate_translate. Please consider using EMData::transform() instead " << endl;
//	Transform3D t(Vec3f(dx, dy, dz), az, alt, phi, Vec3f(pdx,pdy,pdz));
//	rotate_translate(t);

	Transform t;
	t.set_pre_trans(Vec3f(dx, dy, dz));
	t.set_rotation(Dict("type", "eman", "az", az, "alt", alt, "phi", phi));
	t.set_trans(pdx, pdy, pdz);
	rotate_translate(t);
}

//void EMData::rotate_translate(const Transform3D & RA)
//{
//	cout << "Deprecation warning in EMData::rotate_translate. Please consider using EMData::transform() instead " << endl;
//	ENTERFUNC;
//
//#if EMDATA_EMAN2_DEBUG
//	std::cout << "start rotate_translate..." << std::endl;
//#endif
//
//	float scale       = RA.get_scale();
// 	Vec3f dcenter     = RA.get_center();
//	Vec3f translation = RA.get_posttrans();
//	Dict rotation      = RA.get_rotation(Transform3D::EMAN);
////	Transform3D mx = RA;
//	Transform3D RAInv = RA.inverse(); // We're rotating the coordinate system, not the data
//// 	RAInv.printme();
//#if EMDATA_EMAN2_DEBUG
//	vector<string> keys = rotation.keys();
//	vector<string>::const_iterator it;
//	for(it=keys.begin(); it!=keys.end(); ++it) {
////		std::cout << *it << " : " << rotation[*it] << std::endl;
//		std::cout << *it << " : " << (float)rotation.get(*it) << std::endl;
//	}
//#endif
//	float inv_scale = 1.0f;
//
//	if (scale != 0) {
//		inv_scale = 1.0f / scale;
//	}
//
//	float *src_data = 0;
//	float *des_data = 0;
//
//	src_data = get_data();
//	des_data = (float *) EMUtil::em_malloc(nx * ny * nz * sizeof(float));
//
//	if (nz == 1) {
//		float x2c =  nx / 2 - dcenter[0] + RAInv[0][3];
//		float y2c =  ny / 2 - dcenter[1] + RAInv[1][3];
//		float y   = -ny / 2 + dcenter[1]; // changed 0 to 1 in dcenter and below
//		for (int j = 0; j < ny; j++, y += 1.0f) {
//			float x = -nx / 2 + dcenter[0];
//			for (int i = 0; i < nx; i++, x += 1.0f) {
//				float x2 = RAInv[0][0]*x + RAInv[0][1]*y + x2c;
//				float y2 = RAInv[1][0]*x + RAInv[1][1]*y + y2c;
//
//				if (x2 < 0 || x2 >= nx || y2 < 0 || y2 >= ny ) {
//					des_data[i + j * nx] = 0; // It may be tempting to set this value to the
//					// mean but in fact this is not a good thing to do. Talk to S.Ludtke about it.
//				}
//				else {
//					int ii = Util::fast_floor(x2);
//					int jj = Util::fast_floor(y2);
//					int k0 = ii + jj * nx;
//					int k1 = k0 + 1;
//					int k2 = k0 + nx;
//					int k3 = k0 + nx + 1;
//
//					if (ii == nx - 1) {
//						k1--;
//						k3--;
//					}
//					if (jj == ny - 1) {
//						k2 -= nx;
//						k3 -= nx;
//					}
//
//					float t = x2 - ii;
//					float u = y2 - jj;
//
//					des_data[i + j * nx] = Util::bilinear_interpolate(src_data[k0],src_data[k1], src_data[k2], src_data[k3],t,u); // This is essentially basic interpolation
//				}
//			}
//		}
//	}
//	else {
//#if EMDATA_EMAN2_DEBUG
//		std::cout << "This is the 3D case." << std::endl    ;
//#endif
//
//		Transform3D mx = RA;
//		mx.set_scale(inv_scale);
//
//#if EMDATA_EMAN2_DEBUG
////		std::cout << v4[0] << " " << v4[1]<< " " << v4[2]<< " "
////			<< dcenter2[0]<< " "<< dcenter2[1]<< " "<< dcenter2[2] << std::endl;
//#endif
//
//		int nxy = nx * ny;
//		int l = 0;
//
//		float x2c =  nx / 2 - dcenter[0] + RAInv[0][3];;
//		float y2c =  ny / 2 - dcenter[1] + RAInv[1][3];;
//		float z2c =  nz / 2 - dcenter[2] + RAInv[2][3];;
//
//		float z   = -nz / 2 + dcenter[2]; //
//
//		size_t ii, k0, k1, k2, k3, k4, k5, k6, k7;
//		for (int k = 0; k < nz; k++, z += 1.0f) {
//			float y   = -ny / 2 + dcenter[1]; //
//			for (int j = 0; j < ny; j++,   y += 1.0f) {
//				float x = -nx / 2 + dcenter[0];
//				for (int i = 0; i < nx; i++, l++ ,  x += 1.0f) {
//					float x2 = RAInv[0][0] * x + RAInv[0][1] * y + RAInv[0][2] * z + x2c;
//					float y2 = RAInv[1][0] * x + RAInv[1][1] * y + RAInv[1][2] * z + y2c;
//					float z2 = RAInv[2][0] * x + RAInv[2][1] * y + RAInv[2][2] * z + z2c;
//
//
//					if (x2 < 0 || y2 < 0 || z2 < 0 ||
//						x2 >= nx  || y2 >= ny  || z2>= nz ) {
//						des_data[l] = 0;
//					}
//					else {
//						int ix = Util::fast_floor(x2);
//						int iy = Util::fast_floor(y2);
//						int iz = Util::fast_floor(z2);
//						float tuvx = x2-ix;
//						float tuvy = y2-iy;
//						float tuvz = z2-iz;
//						ii = ix + iy * nx + iz * nxy;
//
//						k0 = ii;
//						k1 = k0 + 1;
//						k2 = k0 + nx;
//						k3 = k0 + nx+1;
//						k4 = k0 + nxy;
//						k5 = k1 + nxy;
//						k6 = k2 + nxy;
//						k7 = k3 + nxy;
//
//						if (ix == nx - 1) {
//							k1--;
//							k3--;
//							k5--;
//							k7--;
//						}
//						if (iy == ny - 1) {
//							k2 -= nx;
//							k3 -= nx;
//							k6 -= nx;
//							k7 -= nx;
//						}
//						if (iz == nz - 1) {
//							k4 -= nxy;
//							k5 -= nxy;
//							k6 -= nxy;
//							k7 -= nxy;
//						}
//
//						des_data[l] = Util::trilinear_interpolate(src_data[k0],
//							  src_data[k1],
//							  src_data[k2],
//							  src_data[k3],
//							  src_data[k4],
//							  src_data[k5],
//							  src_data[k6],
//							  src_data[k7],
//							  tuvx, tuvy, tuvz);
//#if EMDATA_EMAN2_DEBUG
//						printf(" ix=%d \t iy=%d \t iz=%d \t value=%f \n", ix ,iy, iz, des_data[l] );
//						std::cout << src_data[ii] << std::endl;
//#endif
//					}
//				}
//			}
//		}
//	}
//
//	if( rdata )
//	{
//		EMUtil::em_free(rdata);
//		rdata = 0;
//	}
//	rdata = des_data;
//
//	scale_pixel(inv_scale);
//
//	attr_dict["origin_x"] = (float) attr_dict["origin_x"] * inv_scale;
//	attr_dict["origin_y"] = (float) attr_dict["origin_y"] * inv_scale;
//	attr_dict["origin_z"] = (float) attr_dict["origin_z"] * inv_scale;
//
//	update();
//	all_translation += translation;
//	EXITFUNC;
//}




void EMData::rotate_x(int dx)
{
	ENTERFUNC;

	if (get_ndim() > 2) {
		throw ImageDimensionException("no 3D image");
	}


	size_t row_size = nx * sizeof(float);
	float *tmp = (float*)EMUtil::em_malloc(row_size);
	float * data = get_data();

	for (int y = 0; y < ny; y++) {
		int y_nx = y * nx;
		for (int x = 0; x < nx; x++) {
			tmp[x] = data[y_nx + (x + dx) % nx];
		}
		EMUtil::em_memcpy(&data[y_nx], tmp, row_size);
	}

	update();
	if( tmp )
	{
		delete[]tmp;
		tmp = 0;
	}
	EXITFUNC;
}

double EMData::dot_rotate_translate(EMData * with, float dx, float dy, float da, const bool mirror)
{
	ENTERFUNC;

	if (!EMUtil::is_same_size(this, with)) {
		LOGERR("images not same size");
		throw ImageFormatException("images not same size");
	}

	if (get_ndim() == 3) {
		LOGERR("1D/2D Images only");
		throw ImageDimensionException("1D/2D only");
	}

	float *this_data = 0;

	this_data = get_data();

	float da_rad = da*(float)M_PI/180.0f;

	float *with_data = with->get_data();
	float mx0 = cos(da_rad);
	float mx1 = sin(da_rad);
	float y = -ny / 2.0f;
	float my0 = mx0 * (-nx / 2.0f - 1.0f) + nx / 2.0f - dx;
	float my1 = -mx1 * (-nx / 2.0f - 1.0f) + ny / 2.0f - dy;
	double result = 0;

	for (int j = 0; j < ny; j++) {
		float x2 = my0 + mx1 * y;
		float y2 = my1 + mx0 * y;

		int ii = Util::fast_floor(x2);
		int jj = Util::fast_floor(y2);
		float t = x2 - ii;
		float u = y2 - jj;

		for (int i = 0; i < nx; i++) {
			t += mx0;
			u -= mx1;

			if (t >= 1.0f) {
				ii++;
				t -= 1.0f;
			}

			if (u >= 1.0f) {
				jj++;
				u -= 1.0f;
			}

			if (t < 0) {
				ii--;
				t += 1.0f;
			}

			if (u < 0) {
				jj--;
				u += 1.0f;
			}

			if (ii >= 0 && ii <= nx - 2 && jj >= 0 && jj <= ny - 2) {
				int k0 = ii + jj * nx;
				int k1 = k0 + 1;
				int k2 = k0 + nx + 1;
				int k3 = k0 + nx;

				float tt = 1 - t;
				float uu = 1 - u;
				int idx = i + j * nx;
				if (mirror) idx = nx-1-i+j*nx; // mirroring of Transforms is always about the y axis
				result += (this_data[k0] * tt * uu + this_data[k1] * t * uu +
						   this_data[k2] * t * u + this_data[k3] * tt * u) * with_data[idx];
			}
		}
		y += 1.0f;
	}

	EXITFUNC;
	return result;
}


EMData *EMData::little_big_dot(EMData * with, bool do_sigma)
{
	ENTERFUNC;

	if (get_ndim() > 2) {
		throw ImageDimensionException("1D/2D only");
	}

	EMData *ret = copy_head();
	ret->set_size(nx,ny,nz);
	ret->to_zero();

	int nx2 = with->get_xsize();
	int ny2 = with->get_ysize();
	float em = with->get_edge_mean();

	float *data = get_data();
	float *with_data = with->get_data();
	float *ret_data = ret->get_data();

	float sum2 = (Util::square((float)with->get_attr("sigma")) +
				  Util::square((float)with->get_attr("mean")));
	if (do_sigma) {
		for (int j = ny2 / 2; j < ny - ny2 / 2; j++) {
			for (int i = nx2 / 2; i < nx - nx2 / 2; i++) {
				float sum = 0;
				float sum1 = 0;
				float summ = 0;
				int k = 0;

				for (int jj = j - ny2 / 2; jj < j + ny2 / 2; jj++) {
					for (int ii = i - nx2 / 2; ii < i + nx2 / 2; ii++) {
						int l = ii + jj * nx;
						sum1 += Util::square(data[l]);
						summ += data[l];
						sum += data[l] * with_data[k];
						k++;
					}
				}
				float tmp_f1 = (sum1 / 2.0f - sum) / (nx2 * ny2);
				float tmp_f2 = Util::square((float)with->get_attr("mean") -
											summ / (nx2 * ny2));
				ret_data[i + j * nx] = sum2 + tmp_f1 - tmp_f2;
			}
		}
	}
	else {
		for (int j = ny2 / 2; j < ny - ny2 / 2; j++) {
			for (int i = nx2 / 2; i < nx - nx2 / 2; i++) {
				float eml = 0;
				float dot = 0;
				float dot2 = 0;

				for (int ii = i - nx2 / 2; ii < i + nx2 / 2; ii++) {
					eml += data[ii + (j - ny2 / 2) * nx] + data[ii + (j + ny2 / 2 - 1) * nx];
				}

				for (int jj = j - ny2 / 2; jj < j + ny2 / 2; jj++) {
					eml += data[i - nx2 / 2 + jj * nx] + data[i + nx2 / 2 - 1 + jj * nx];
				}

				eml /= (nx2 + ny2) * 2.0f;
				int k = 0;

				for (int jj = j - ny2 / 2; jj < j + ny2 / 2; jj++) {
					for (int ii = i - nx2 / 2; ii < i + nx2 / 2; ii++) {
						dot += (data[ii + jj * nx] - eml) * (with_data[k] - em);
						dot2 += Util::square(data[ii + jj * nx] - eml);
						k++;
					}
				}

				dot2 = std::sqrt(dot2);

				if (dot2 == 0) {
					ret_data[i + j * nx] = 0;
				}
				else {
					ret_data[i + j * nx] = dot / (nx2 * ny2 * dot2 * (float)with->get_attr("sigma"));
				}
			}
		}
	}

	ret->update();

	EXITFUNC;
	return ret;
}


EMData *EMData::do_radon()
{
	ENTERFUNC;

	if (get_ndim() != 2) {
		throw ImageDimensionException("2D only");
	}

	if (nx != ny) {
		throw ImageFormatException("square image only");
	}

	EMData *result = new EMData();
	result->set_size(nx, ny, 1);
	result->to_zero();
	float *result_data = result->get_data();

	EMData *this_copy = this;
	this_copy = copy();

	for (int i = 0; i < nx; i++) {
		Transform t(Dict("type","2d","alpha",(float) M_PI * 2.0f * i / nx));
		this_copy->transform(t);

		float *copy_data = this_copy->get_data();

		for (int y = 0; y < nx; y++) {
			for (int x = 0; x < nx; x++) {
				if (Util::square(x - nx / 2) + Util::square(y - nx / 2) <= nx * nx / 4) {
					result_data[i + y * nx] += copy_data[x + y * nx];
				}
			}
		}

		this_copy->update();
	}

	result->update();

	if( this_copy )
	{
		delete this_copy;
		this_copy = 0;
	}

	EXITFUNC;
	return result;
}

void EMData::zero_corner_circulant(const int radius)
{
	if ( nz > 1 && nz < (2*radius+1) ) throw ImageDimensionException("Error: cannot zero corner - nz is too small");
	if ( ny > 1 && ny < (2*radius+1) ) throw ImageDimensionException("Error: cannot zero corner - ny is too small");
	if ( nx > 1 && nx < (2*radius+1) ) throw ImageDimensionException("Error: cannot zero corner - nx is too small");

	int it_z = radius;
	int it_y = radius;
	int it_x = radius;

	if ( nz == 1 ) it_z = 0;
	if ( ny == 1 ) it_y = 0;
	if ( nx == 1 ) it_z = 0;

	if ( nz == 1 && ny == 1 )
	{
		for ( int x = -it_x; x <= it_x; ++x )
			get_value_at_wrap(x) = 0;

	}
	else if ( nz == 1 )
	{
		for ( int y = -it_y; y <= it_y; ++y)
			for ( int x = -it_x; x <= it_x; ++x )
				get_value_at_wrap(x,y) = 0;
	}
	else
	{
		for( int z = -it_z; z <= it_z; ++z )
			for ( int y = -it_y; y <= it_y; ++y)
				for ( int x = -it_x; x < it_x; ++x )
					get_value_at_wrap(x,y,z) = 0;

	}

}

EMData *EMData::calc_ccf(EMData * with, fp_flag fpflag,bool center)
{
	ENTERFUNC;

	if( with == 0 ) {
#ifdef EMAN2_USING_CUDA //CUDA 
	if(EMData::usecuda == 1 && cudarwdata) {
		//cout << "calc ccf" << endl;
		EMData* ifft = 0;
		bool delifft = false;
		int offset = 0;
		
		//do fft if not alreay done
		if(!is_complex()){
			ifft = do_fft_cuda();
			delifft = true;
			offset = 2 - nx%2;
		}else{
			ifft = this;
		}
		calc_conv_cuda(ifft->cudarwdata,ifft->cudarwdata,nx + offset, ny, nz); //this is the business end, results go in afft
		
		EMData * conv = ifft->do_ift_cuda();
		if(delifft) delete ifft;
		conv->update();
			
		return conv;
	}
#endif
		EXITFUNC;
		return convolution(this,this,fpflag, center);
	}
	else if ( with == this ){ // this if statement is not necessary, the correlation function tests to see if with == this
		EXITFUNC;
		return correlation(this, this, fpflag,center);
	}
	else {

#ifdef EMAN2_USING_CUDA //CUDA 
		// assume always get rw data (makes life a lot easier!!! 
		// also assume that both images are the same size. When using CUDA we are only interested in speed, not flexibility!!
		// P.S. (I feel like I am pounding square pegs into a round holes with CUDA)
		if(EMData::usecuda == 1 && cudarwdata && with->cudarwdata) {
			//cout << "using CUDA for ccf" << endl;
			EMData* afft = 0;
			EMData* bfft = 0;
			bool delafft = false, delbfft = false;
			int offset = 0;
			
			//do ffts if not alreay done
			if(!is_complex()){
				afft = do_fft_cuda();
				delafft = true;
				offset = 2 - nx%2;
				//cout << "Do cuda FFT A" << endl;
			}else{
				afft = this;
			}
			if(!with->is_complex()){
				bfft = with->do_fft_cuda();
				//cout << "Do cuda FFT B" << endl;
				delbfft = true;
			}else{
				bfft = with;
			}

			calc_ccf_cuda(afft->cudarwdata,bfft->cudarwdata,nx + offset, ny, nz); //this is the business end, results go in afft
			
			if(delbfft) delete bfft;
			
			EMData * corr = afft->do_ift_cuda();
			if(delafft) delete afft;
			//cor->do_ift_inplace_cuda();//a bit faster, but I'll alos need to rearrnage the mem structure for it to work, BUT this is very SLOW.
			corr->update();
			
			return corr;
		}
#endif
		
		// If the argument EMData pointer is not the same size we automatically resize it
		bool undoresize = false;
		int wnx = with->get_xsize(); int wny = with->get_ysize(); int wnz = with->get_zsize();
		if (!(is_complex()^with->is_complex()) && (wnx != nx || wny != ny || wnz != nz) ) {
			Region r((wnx-nx)/2, (wny-ny)/2, (wnz-nz)/2,nx,ny,nz);
			with->clip_inplace(r);
			undoresize = true;
		}

		EMData* cor = correlation(this, with, fpflag, center);

		// If the argument EMData pointer was resized, it is returned to its original dimensions
		if ( undoresize ) {
			Region r((nx-wnx)/2, (ny-wny)/2,(nz-wnz)/2,wnx,wny,wnz);
			with->clip_inplace(r);
		}

		EXITFUNC;
		return cor;
	}
}

EMData *EMData::calc_ccfx( EMData * const with, int y0, int y1, bool no_sum, bool flip,bool usez)
{
	ENTERFUNC;

	if (!with) {
		LOGERR("NULL 'with' image. ");
		throw NullPointerException("NULL input image");
	}

	if (!EMUtil::is_same_size(this, with)) {
		LOGERR("images not same size: (%d,%d,%d) != (%d,%d,%d)",
			   nx, ny, nz,
			   with->get_xsize(), with->get_ysize(), with->get_zsize());
		throw ImageFormatException("images not same size");
	}
	if (get_ndim() > 2) {
		LOGERR("2D images only");
		throw ImageDimensionException("2D images only");
	}

	if (y1 <= y0) {
		y1 = ny;
	}

	if (y0 >= y1) {
		y0 = 0;
	}

	if (y0 < 0) {
		y0 = 0;
	}

	if (y1 > ny) {
		y1 = ny;
	}
	if (is_complex_x() || with->is_complex_x() ) throw; // Woops don't support this anymore!

	static int nx_fft = 0;
	static int ny_fft = 0;
	static EMData f1;
	static EMData f2;
	static EMData rslt;

	int height = y1-y0;
	int width = (nx+2-(nx%2));
	int wpad = ((width+3)/4)*4;			// This is for 128 bit alignment of rows to prevent SSE crashes
	if (wpad != nx_fft || height != ny_fft ) {	// Seems meaningless, but due to static definitions above. f1,f2 are cached to prevent multiple reallocations
		f1.set_size(wpad,height);
		f2.set_size(wpad,height);
		rslt.set_size(nx,height);
		nx_fft = wpad;
		ny_fft = height;
	}

#ifdef EMAN2_USING_CUDA
	// FIXME : Not tested with new wpad change
	if (EMData::usecuda == 1 && cudarwdata && with->cudarwdata) {
		//cout << "calc_ccfx CUDA" << endl;
		if(!f1.cudarwdata) f1.rw_alloc();
		if(!f2.cudarwdata) f2.rw_alloc();
		if(!rslt.cudarwdata) rslt.rw_alloc();
		cuda_dd_fft_real_to_complex_nd(cudarwdata, f1.cudarwdata, nx, 1, 1, height);
		cuda_dd_fft_real_to_complex_nd(with->cudarwdata, f2.cudarwdata, nx, 1, 1, height);
		calc_ccf_cuda(f1.cudarwdata, f2.cudarwdata, nx, ny, nz);
		cuda_dd_fft_complex_to_real_nd(f1.cudarwdata, rslt.cudarwdata, nx, 1, 1, height);
		if(no_sum){
			EMData* result = new EMData(rslt);
			return result;
		}
		EMData* cf = new EMData(0,0,nx,1,1); //cuda constructor
		cf->runcuda(emdata_column_sum(rslt.cudarwdata, nx, ny));
		cf->update();
		
		EXITFUNC;
		return cf;
	}
#endif

//	printf("%d %d %d\n",(int)get_attr("nx"),(int)f2.get_attr("nx"),width);

	float *d1 = get_data();
	float *d2 = with->get_data();
	float *f1d = f1.get_data();
	float *f2d = f2.get_data();
	for (int j = 0; j < height; j++) {
		EMfft::real_to_complex_1d(d1 + j * nx, f1d+j*wpad, nx);
		EMfft::real_to_complex_1d(d2 + j * nx, f2d+j*wpad, nx);
	}

	if(flip == false) {
		for (int j = 0; j < height; j++) {
			float *f1a = f1d + j * wpad;
			float *f2a = f2d + j * wpad;

			for (int i = 0; i < width / 2; i++) {
				float re1 = f1a[2*i];
				float re2 = f2a[2*i];
				float im1 = f1a[2*i+1];
				float im2 = f2a[2*i+1];

				f1d[j*wpad+i*2] = re1 * re2 + im1 * im2;
				f1d[j*wpad+i*2+1] = im1 * re2 - re1 * im2;
			}
		}
	} else {
		for (int j = 0; j < height; j++) {
			float *f1a = f1d + j * wpad;
			float *f2a = f2d + j * wpad;

			for (int i = 0; i < width / 2; i++) {
				float re1 = f1a[2*i];
				float re2 = f2a[2*i];
				float im1 = f1a[2*i+1];
				float im2 = f2a[2*i+1];

				f1d[j*wpad+i*2] = re1 * re2 - im1 * im2;
				f1d[j*wpad+i*2+1] = im1 * re2 + re1 * im2;
			}
		}
	}

	float* rd = rslt.get_data();
	for (int j = y0; j < y1; j++) {
		EMfft::complex_to_real_1d(f1d+j*wpad, rd+j*nx, nx);
	}

	// This converts the CCF values to Z values (in terms of standard deviations above the mean), on a per-row basis
	// The theory is that this should better weight the radii that contribute most strongly to the orientation determination
	if (usez) {
		for (int y=0; y<height; y++) {
			float mn=0,sg=0;
			for (int x=0; x<nx; x++) {
				mn+=rd[x+y*nx];
				sg+=rd[x+y*nx]*rd[x+y*nx];
			}
			mn/=(float)nx;						//mean
			sg=std::sqrt(sg/(float)nx-mn*mn);	//sigma
			
			for (int x=0; x<nx; x++) rd[x+y*nx]=(rd[x+y*nx]-mn)/sg;
		}
	}

	if (no_sum) {
		rslt.update(); // This is important in terms of the copy - the returned object won't have the correct flags unless we do this
		EXITFUNC;
		return new EMData(rslt);
	} else {
		EMData *cf = new EMData(nx,1,1);
		cf->to_zero();
		float *c = cf->get_data();
		for (int j = 0; j < height; j++) {
			for(int i = 0; i < nx; ++i) {
				c[i] += rd[i+j*nx];
			}
		}
		cf->update();
		EXITFUNC;
		return cf;
	}
}

EMData *EMData::make_rotational_footprint_cmc( bool unwrap) {
	ENTERFUNC;
	update_stat();
	// Note that rotational_footprint caching saves a large amount of time
	// but this is at the expense of memory. Note that a policy is hardcoded here,
	// that is that caching is only employed when premasked is false and unwrap
	// is true - this is probably going to be what is used in most scenarios
	// as advised by Steve Ludtke - In terms of performance this caching doubles the metric
	// generated by e2speedtest.
	if ( rot_fp != 0 && unwrap == true) {
		return new EMData(*rot_fp);
	}

	static EMData obj_filt;
	EMData* filt = &obj_filt;
	filt->set_complex(true);


	// The filter object is nothing more than a cached high pass filter
	// Ultimately it is used an argument to the EMData::mult(EMData,prevent_complex_multiplication (bool))
	// function in calc_mutual_correlation. Note that in the function the prevent_complex_multiplication
	// set to true, which is used for speed reasons.
	if (filt->get_xsize() != nx+2-(nx%2) || filt->get_ysize() != ny ||
		   filt->get_zsize() != nz ) {
		filt->set_size(nx+2-(nx%2), ny, nz);
		filt->to_one();

		filt->process_inplace("filter.highpass.gauss", Dict("cutoff_abs", 1.5f/nx));
	}

	EMData *ccf = this->calc_mutual_correlation(this, true,filt);
	ccf->sub(ccf->get_edge_mean());
	EMData *result = ccf->unwrap();
	delete ccf; ccf = 0;

	EXITFUNC;
	if ( unwrap == true)
	{
	// this if statement reflects a strict policy of caching in only one scenario see comments at beginning of function block

// Note that the if statement at the beginning of this function ensures that rot_fp is not zero, so there is no need
// to throw any exception
// if ( rot_fp != 0 ) throw UnexpectedBehaviorException("The rotational foot print is only expected to be cached if it is not NULL");

// Here is where the caching occurs - the rot_fp takes ownsherhip of the pointer, and a deep copied EMData object is returned.
// The deep copy invokes a cost in terms of CPU cycles and memory, but prevents the need for complicated memory management (reference counting)
		rot_fp = result;
		return new EMData(*rot_fp);
	}
	else return result;
}

EMData *EMData::make_rotational_footprint( bool unwrap) {
	ENTERFUNC;
	update_stat();
	// Note that rotational_footprint caching saves a large amount of time
	// but this is at the expense of memory. Note that a policy is hardcoded here,
	// that is that caching is only employed when premasked is false and unwrap
	// is true - this is probably going to be what is used in most scenarios
	// as advised by Steve Ludtke - In terms of performance this caching doubles the metric
	// generated by e2speedtest.
	if ( rot_fp != 0 && unwrap == true) {
		return new EMData(*rot_fp);
	}

	EMData* ccf = this->calc_ccf(this,CIRCULANT,true);
//	EMData* ccf = this->calc_ccf(this,PADDED,true);		# this would probably be a bit better, but takes 4x longer  :^/
	ccf->sub(ccf->get_edge_mean());
	//ccf->process_inplace("xform.phaseorigin.tocenter"); ccf did the centering
	EMData *result = ccf->unwrap();
	delete ccf; ccf = 0;

	EXITFUNC;
	if ( unwrap == true)
	{ // this if statement reflects a strict policy of caching in only one scenario see comments at beginning of function block

// Note that the if statement at the beginning of this function ensures that rot_fp is not zero, so there is no need
// to throw any exception
// if ( rot_fp != 0 ) throw UnexpectedBehaviorException("The rotational foot print is only expected to be cached if it is not NULL");

// Here is where the caching occurs - the rot_fp takes ownsherhip of the pointer, and a deep copied EMData object is returned.
// The deep copy invokes a cost in terms of CPU cycles and memory, but prevents the need for complicated memory management (reference counting)
		rot_fp = result;
		return new EMData(*rot_fp);
	}
	else return result;
}

EMData *EMData::make_rotational_footprint_e1( bool unwrap)
{
	ENTERFUNC;

	update_stat();
	// Note that rotational_footprint caching saves a large amount of time
	// but this is at the expense of memory. Note that a policy is hardcoded here,
	// that is that caching is only employed when premasked is false and unwrap
	// is true - this is probably going to be what is used in most scenarios
	// as advised by Steve Ludtke - In terms of performance this caching doubles the metric
	// generated by e2speedtest.
	if ( rot_fp != 0 && unwrap == true) {
		return new EMData(*rot_fp);
	}

	static EMData obj_filt;
	EMData* filt = &obj_filt;
	filt->set_complex(true);
// 	Region filt_region;

// 	if (nx & 1) {
// 		LOGERR("even image xsize only");		throw ImageFormatException("even image xsize only");
// 	}

	int cs = (((nx * 7 / 4) & 0xfffff8) - nx) / 2; // this pads the image to 1 3/4 * size with result divis. by 8

	static EMData big_clip;
	int big_x = nx+2*cs;
	int big_y = ny+2*cs;
	int big_z = 1;
	if ( nz != 1 ) {
		big_z = nz+2*cs;
	}


	if ( big_clip.get_xsize() != big_x || big_clip.get_ysize() != big_y || big_clip.get_zsize() != big_z ) {
		big_clip.set_size(big_x,big_y,big_z);
	}
	// It is important to set all newly established pixels around the boundaries to the mean
	// If this is not done then the associated rotational alignment routine breaks, in fact
	// everythin just goes foo.

	big_clip.to_value(get_edge_mean());

	if (nz != 1) {
		big_clip.insert_clip(this,IntPoint(cs,cs,cs));
	} else  {
		big_clip.insert_clip(this,IntPoint(cs,cs,0));
	}
	
	// The filter object is nothing more than a cached high pass filter
	// Ultimately it is used an argument to the EMData::mult(EMData,prevent_complex_multiplication (bool))
	// function in calc_mutual_correlation. Note that in the function the prevent_complex_multiplication
	// set to true, which is used for speed reasons.
	if (filt->get_xsize() != big_clip.get_xsize() +2-(big_clip.get_xsize()%2) || filt->get_ysize() != big_clip.get_ysize() ||
		   filt->get_zsize() != big_clip.get_zsize()) {
		filt->set_size(big_clip.get_xsize() + 2-(big_clip.get_xsize()%2), big_clip.get_ysize(), big_clip.get_zsize());
		filt->to_one();
		filt->process_inplace("filter.highpass.gauss", Dict("cutoff_abs", 1.5f/nx));
#ifdef EMAN2_USING_CUDA
		/*
		if(EMData::usecuda == 1 && big_clip.cudarwdata)
		{
			filt->copy_to_cuda(); // since this occurs just once for many images, we don't pay much of a speed pentalty here, and we avoid the hassel of messing with sparx
		}
		*/
#endif
	}
#ifdef EMAN2_USING_CUDA
	/*
	if(EMData::usecuda == 1 && big_clip.cudarwdata && !filt->cudarwdata)
	{
		filt->copy_to_cuda(); // since this occurs just once for many images, we don't pay much of a speed pentalty here, and we avoid the hassel of messing with sparx
	}
	*/
#endif
	
	EMData *mc = big_clip.calc_mutual_correlation(&big_clip, true,filt);
 	mc->sub(mc->get_edge_mean());

	static EMData sml_clip;
	int sml_x = nx * 3 / 2;
	int sml_y = ny * 3 / 2;
	int sml_z = 1;
	if ( nz != 1 ) {
		sml_z = nz * 3 / 2;
	}

	if ( sml_clip.get_xsize() != sml_x || sml_clip.get_ysize() != sml_y || sml_clip.get_zsize() != sml_z ) {
		sml_clip.set_size(sml_x,sml_y,sml_z);	}
	if (nz != 1) {
		sml_clip.insert_clip(mc,IntPoint(-cs+nx/4,-cs+ny/4,-cs+nz/4));
	} else {
		sml_clip.insert_clip(mc,IntPoint(-cs+nx/4,-cs+ny/4,0));		// same as padding change above
	}
		
	delete mc; mc = 0;
	EMData * result = NULL;
	
	if (nz == 1) {
		if (!unwrap) {
#ifdef EMAN2_USING_CUDA
			//if(EMData::usecuda == 1 && sml_clip.cudarwdata) throw UnexpectedBehaviorException("shap masking is not yet supported by CUDA");
#endif
			result = sml_clip.process("mask.sharp", Dict("outer_radius", -1, "value", 0));

		}
		else {
			result = sml_clip.unwrap();
		}
	}
	else {
		// I am not sure why there is any consideration of non 2D images, but it was here
		// in the first port so I kept when I cleaned this function up (d.woolford)
// 		result = clipped_mc;
		result = new EMData(sml_clip);
	}
	
#ifdef EMAN2_USING_CUDA
	//if (EMData::usecuda == 1) sml_clip.roneedsanupdate(); //If we didn't do this then unwrap would use data from the previous call of this function, happens b/c sml_clip is static
#endif
	EXITFUNC;
	if ( unwrap == true)
	{ // this if statement reflects a strict policy of caching in only one scenario see comments at beginning of function block

		// Note that the if statement at the beginning of this function ensures that rot_fp is not zero, so there is no need
		// to throw any exception
		if ( rot_fp != 0 ) throw UnexpectedBehaviorException("The rotational foot print is only expected to be cached if it is not NULL");

		// Here is where the caching occurs - the rot_fp takes ownsherhip of the pointer, and a deep copied EMData object is returned.
		// The deep copy invokes a cost in terms of CPU cycles and memory, but prevents the need for complicated memory management (reference counting)
		rot_fp = result;
		return new EMData(*rot_fp);
	}
	else return result;
}

EMData *EMData::make_footprint(int type)
{
//	printf("Make fp %d\n",type);
	if (type==0) {
		EMData *un=make_rotational_footprint_e1(); // Use EMAN1's footprint strategy
		if (un->get_ysize() <= 6) {
			throw UnexpectedBehaviorException("In EMData::make_footprint. The rotational footprint is too small");
		}
		EMData *tmp=un->get_clip(Region(0,4,un->get_xsize(),un->get_ysize()-6));	// 4 and 6 are empirical
		EMData *cx=tmp->calc_ccfx(tmp,0,-1,1);
		EMData *fp=cx->get_clip(Region(0,0,cx->get_xsize()/2,cx->get_ysize()));
		delete un;
		delete tmp;
		delete cx;
		return fp;
	}
	else if (type==1 || type==2 ||type==5 || type==6) {
		int i,j,kx,ky,lx,ly;

		EMData *fft=do_fft();

		// map for x,y -> radius for speed
		int rmax=(get_xsize()+1)/2;
		float *rmap=(float *)malloc(rmax*rmax*sizeof(float));
		for (i=0; i<rmax; i++) {
			for (j=0; j<rmax; j++) {
#ifdef _WIN32
				rmap[i+j*rmax]=_hypotf((float)i,(float)j);
#else
				rmap[i+j*rmax]=hypot((float)i,(float)j);
#endif	//_WIN32
//				printf("%d\t%d\t%f\n",i,j,rmap[i+j*rmax]);
			}
		}

		EMData *fp=new EMData(rmax*2+2,rmax*2,1);
		fp->set_complex(1);
		fp->to_zero();

		// Two vectors in to complex space (kx,ky) and (lx,ly)
		// We are computing the bispectrum, f(k).f(l).f*(k+l)
		// but integrating out two dimensions, leaving |k|,|l|
		for (kx=-rmax+1; kx<rmax; kx++) {
			for (ky=-rmax+1; ky<rmax; ky++) {
				for (lx=-rmax+1; lx<rmax; lx++) {
					for (ly=-rmax+1; ly<rmax; ly++) {
						int ax=kx+lx;
						int ay=ky+ly;
						if (abs(ax)>=rmax || abs(ay)>=rmax) continue;
						int r1=(int)floor(.5+rmap[abs(kx)+rmax*abs(ky)]);
						int r2=(int)floor(.5+rmap[abs(lx)+rmax*abs(ly)]);
//						if (r1>500 ||r2>500) printf("%d\t%d\t%d\t%d\t%d\t%d\n",kx,ky,lx,ly,r1,r2);
//						float r3=rmap[ax+rmax*ay];
						if (r1+r2>=rmax) continue;

						std::complex<float> p=fft->get_complex_at(kx,ky)*fft->get_complex_at(lx,ly)*conj(fft->get_complex_at(ax,ay));
						fp->set_value_at(r1*2,r2,p.real()+fp->get_value_at(r1*2,r2));		// We keep only the real component in anticipation of zero phase sum
//						fp->set_value_at(r1*2,rmax*2-r2-1,  fp->get_value_at(r1*2,r2));		// We keep only the real component in anticipation of zero phase sum
//						fp->set_value_at(r1*2+1,r2,p.real()+fp->get_value_at(r1*2+1,r2));		// We keep only the real component in anticipation of zero phase sum
						fp->set_value_at(r1*2+1,r2,fp->get_value_at(r1*2+1,r2)+1);			// a normalization counter
					}
				}
			}
		}

		// Normalizes the pixels based on the accumulated counts then sets the imaginary components back to zero
		if (type==5 || type==6) {
			for (i=0; i<rmax*2; i+=2) {
				for (j=0; j<rmax; j++) {
					float norm=fp->get_value_at(i+1,j);
#ifdef _WIN32
					fp->set_value_at(i,rmax*2-j-1,pow(fp->get_value_at(i,j)/(norm==0.0f?1.0f:norm), 1.0f/3.0f));
					fp->set_value_at(i,j,pow(fp->get_value_at(i,j)/(norm==0.0f?1.0f:norm), 1.0f/3.0f));
#else
					fp->set_value_at(i,rmax*2-j-1,cbrt(fp->get_value_at(i,j)/(norm==0?1.0:norm)));
					fp->set_value_at(i,j,cbrt(fp->get_value_at(i,j)/(norm==0?1.0:norm)));
#endif	//_WIN32
					fp->set_value_at(i+1,j,0.0);
				}
			}
		}
		else {
			for (i=0; i<rmax*2; i+=2) {
				for (j=0; j<rmax; j++) {
					float norm=fp->get_value_at(i+1,j);
					fp->set_value_at(i,rmax*2-j-1,fp->get_value_at(i,j)/(norm==0.0f?1.0f:norm));
					fp->set_value_at(i,j,fp->get_value_at(i,j)/(norm==0.0f?1.0f:norm));
					fp->set_value_at(i+1,j,0.0);
				}
			}
		}

		free(rmap);
		if (type==2||type==6) {
			EMData *f2=fp->do_ift();
			if (f2->get_value_at(0,0)<0) f2->mult(-1.0f);
			f2->process_inplace("xform.phaseorigin.tocorner");
			delete fp;
			return f2;
		}
		return fp;
	}
	else if (type==3 || type==4) {
		int h,i,j,kx,ky,lx,ly;

		EMData *fft=do_fft();

		// map for x,y -> radius for speed
		int rmax=(get_xsize()+1)/2;
		float *rmap=(float *)malloc(rmax*rmax*sizeof(float));
		for (i=0; i<rmax; i++) {
			for (j=0; j<rmax; j++) {
#ifdef _WIN32
				rmap[i+j*rmax]=_hypotf((float)i,(float)j);
#else
				rmap[i+j*rmax]=hypot((float)i,(float)j);
#endif	//_WIN32
//				printf("%d\t%d\t%f\n",i,j,rmap[i+j*rmax]);
			}
		}

		EMData *fp=new EMData(rmax*2+2,rmax*2,16);

		fp->set_complex(1);
		fp->to_zero();

		// Two vectors in to complex space (kx,ky) and (lx,ly)
		// We are computing the bispectrum, f(k).f(l).f*(k+l)
		// but integrating out two dimensions, leaving |k|,|l|
		for (kx=-rmax+1; kx<rmax; kx++) {
			for (ky=-rmax+1; ky<rmax; ky++) {
				for (lx=-rmax+1; lx<rmax; lx++) {
					for (ly=-rmax+1; ly<rmax; ly++) {
						int ax=kx+lx;
						int ay=ky+ly;
						if (abs(ax)>=rmax || abs(ay)>=rmax) continue;
						float rr1=rmap[abs(kx)+rmax*abs(ky)];
						float rr2=rmap[abs(lx)+rmax*abs(ly)];
						int r1=(int)floor(.5+rr1);
						int r2=(int)floor(.5+rr2);
//						if (r1>500 ||r2>500) printf("%d\t%d\t%d\t%d\t%d\t%d\n",kx,ky,lx,ly,r1,r2);
//						float r3=rmap[ax+rmax*ay];
						if (r1+r2>=rmax || rr1==0 ||rr2==0) continue;

						std::complex<float> p=fft->get_complex_at(kx,ky)*fft->get_complex_at(lx,ly)*conj(fft->get_complex_at(ax,ay));
						int dot=(int)floor((kx*lx+ky*ly)/(rr1*rr2)*7.5);					// projection of k on l 0-31
						if (dot<0) dot=16+dot;
//						int dot=(int)floor((kx*lx+ky*ly)/(rr1*rr2)*7.5+8.0);					// projection of k on l 0-15
						fp->set_value_at(r1*2,r2,dot,p.real()+fp->get_value_at(r1*2,r2,dot));		// We keep only the real component in anticipation of zero phase sum
//						fp->set_value_at(r1*2,rmax*2-r2-1,  fp->get_value_at(r1*2,r2));		// We keep only the real component in anticipation of zero phase sum
//						fp->set_value_at(r1*2+1,r2,p.real()+fp->get_value_at(r1*2+1,r2));		// We keep only the real component in anticipation of zero phase sum
						fp->set_value_at(r1*2+1,r2,dot,fp->get_value_at(r1*2+1,r2,dot)+1);			// a normalization counter
					}
				}
			}
		}

		// Normalizes the pixels based on the accumulated counts then sets the imaginary components back to zero
		for (i=0; i<rmax*2; i+=2) {
			for (j=0; j<rmax; j++) {
				for (h=0; h<16; h++) {
					float norm=fp->get_value_at(i+1,j,h);
//					fp->set_value_at(i,rmax*2-j-1,h,cbrt(fp->get_value_at(i,j,h)/(norm==0?1.0:norm)));
//					fp->set_value_at(i,j,h,cbrt(fp->get_value_at(i,j,h)/(norm==0?1.0:norm)));
					fp->set_value_at(i,rmax*2-j-1,h,(fp->get_value_at(i,j,h)/(norm==0.0f?1.0f:norm)));
					fp->set_value_at(i,j,h,(fp->get_value_at(i,j,h)/(norm==0.0f?1.0f:norm)));
	//				fp->set_value_at(i,rmax*2-j-1,fp->get_value_at(i,j)/(norm==0?1.0:norm));
	//				fp->set_value_at(i,j,fp->get_value_at(i,j)/(norm==0?1.0:norm));
					fp->set_value_at(i+1,j,h,0.0);
				}
			}
		}

		free(rmap);
		if (type==4) {
			EMData *f2=fp->do_ift();
			if (f2->get_value_at(0,0,0)<0) f2->mult(-1.0f);
			f2->process_inplace("xform.phaseorigin.tocorner");
			delete fp;
			return f2;
		}
		return fp;
	}
	throw UnexpectedBehaviorException("There is not implementation for the parameters you specified");
}


EMData *EMData::calc_mutual_correlation(EMData * with, bool tocenter, EMData * filter)
{
	ENTERFUNC;

	if (with && !EMUtil::is_same_size(this, with)) {
		LOGERR("images not same size");
		throw ImageFormatException( "images not same size");
	}

#ifdef EMAN2_USING_CUDA
	if(EMData::usecuda == 1 && cudarwdata && with->cudarwdata)
	{	

		EMData* this_fft = do_fft_cuda();

		EMData *cf = 0;
		if (with && with != this) {
			cf = with->do_fft_cuda();
		}else{
			cf = this_fft->copy();
		}
		
		if (filter) {
			if (!EMUtil::is_same_size(filter, cf)) {
				LOGERR("improperly sized filter");
				throw ImageFormatException("improperly sized filter");
			}
			mult_complex_efficient_cuda(cf->cudarwdata, filter->cudarwdata, cf->get_xsize(), cf->get_ysize(), cf->get_zsize(), 1);
			mult_complex_efficient_cuda(this_fft->cudarwdata, filter->cudarwdata, this_fft->get_xsize(), this_fft->get_ysize(), this_fft->get_zsize(), 1);
		}
		
		mcf_cuda(this_fft->cudarwdata, cf->cudarwdata, this_fft->get_xsize(), this_fft->get_ysize(), this_fft->get_zsize());
		
		EMData *f2 = cf->do_ift_cuda();

		if (tocenter) {
			f2->process_inplace("xform.phaseorigin.tocenter");
		}

		if( cf )
		{
			delete cf;
			cf = 0;
		}

		if( this_fft )
		{
			delete this_fft;
			this_fft = 0;
		}

		f2->set_attr("label", "MCF");
		f2->set_path("/tmp/eman.mcf");
		f2->update();

		EXITFUNC;
		return f2;
	}
#endif

	EMData *this_fft = 0;
	this_fft = do_fft();

	if (!this_fft) {

		LOGERR("FFT returns NULL image");
		throw NullPointerException("FFT returns NULL image");
	}

	this_fft->ap2ri(); //this is not needed!
	EMData *cf = 0;

	if (with && with != this) {
		cf = with->do_fft();
		if (!cf) {
			LOGERR("FFT returns NULL image");
			throw NullPointerException("FFT returns NULL image");
		}
		cf->ap2ri(); //nor is this!
	}
	else {
		cf = this_fft->copy();
	}
	
	if (filter) {
		if (!EMUtil::is_same_size(filter, cf)) {
			LOGERR("improperly sized filter");
			throw ImageFormatException("improperly sized filter");
		}
		
		cf->mult_complex_efficient(*filter,true); //insanely this is required....
		this_fft->mult(*filter,true);
		//cf->mult_complex_efficient(*filter,7); // takes advantage of the fact that the filter is 1 everywhere but near the origin
		//this_fft->mult_complex_efficient(*filter,7);
		/*cf->mult_complex_efficient(*filter,5);
		this_fft->mult_complex_efficient(*filter,5);*/
	}

	float *rdata1 = this_fft->get_data();
	float *rdata2 = cf->get_data();
	size_t this_fft_size = (size_t)this_fft->get_xsize() * this_fft->get_ysize() * this_fft->get_zsize();

	if (with == this) {
		for (size_t i = 0; i < this_fft_size; i += 2) {
			rdata2[i] = std::sqrt(rdata1[i] * rdata2[i] + rdata1[i + 1] * rdata2[i + 1]);
			rdata2[i + 1] = 0;
		}

		this_fft->update();
		cf->update();
	}
	else {
		for (size_t i = 0; i < this_fft_size; i += 2) {
			rdata2[i] = (rdata1[i] * rdata2[i] + rdata1[i + 1] * rdata2[i + 1]);
			rdata2[i + 1] = (rdata1[i + 1] * rdata2[i] - rdata1[i] * rdata2[i + 1]);
		}
		
		//This seems like a bug, but it probably is never used....
		for (size_t i = 0; i < this_fft_size; i += 2) {
			float t = Util::square(rdata2[i]) + Util::square(rdata2[i + 1]);
			if (t != 0) {
				t = pow(t, 0.25f);
				rdata2[i] /= t;
				rdata2[i + 1] /= t;
			}
		}
		this_fft->update();
		cf->update();
	}

	EMData *f2 = cf->do_ift();

	if (tocenter) {
		f2->process_inplace("xform.phaseorigin.tocenter");
	}

	if( cf )
	{
		delete cf;
		cf = 0;
	}

	if( this_fft )
	{
		delete this_fft;
		this_fft = 0;
	}

	f2->set_attr("label", "MCF");
	f2->set_path("/tmp/eman.mcf");

	EXITFUNC;
	return f2;
}


vector < float > EMData::calc_hist(int hist_size, float histmin, float histmax,const float& brt, const float& cont)
{
	ENTERFUNC;

	static size_t prime[] = { 1, 3, 7, 11, 17, 23, 37, 59, 127, 253, 511 };

	if (histmin == histmax) {
		histmin = get_attr("minimum");
		histmax = get_attr("maximum");
	}

	vector <float> hist(hist_size, 0.0);

	int p0 = 0;
	int p1 = 0;
	size_t size = (size_t)nx * ny * nz;
	if (size < 300000) {
		p0 = 0;
		p1 = 0;
	}
	else if (size < 2000000) {
		p0 = 2;
		p1 = 3;
	}
	else if (size < 8000000) {
		p0 = 4;
		p1 = 6;
	}
	else {
		p0 = 7;
		p1 = 9;
	}

	if (is_complex() && p0 > 0) {
		p0++;
		p1++;
	}

	size_t di = 0;
//	float norm = 0;
	size_t n = hist.size();

	float * data = get_data();
	for (int k = p0; k <= p1; ++k) {
		if (is_complex()) {
			di = prime[k] * 2;
		}
		else {
			di = prime[k];
		}

//		norm += (float)size / (float) di;
		float w = (float)n / (histmax - histmin);

		for(size_t i=0; i<=size-di; i += di) {
			float val;
			if (cont != 1.0f || brt != 0)val = cont*(data[i]+brt);
			else val = data[i];
			int j = Util::round((val - histmin) * w);
			if (j >= 0 && j < (int) n) {
				hist[j] += 1;
			}
		}
	}
/*
	for (size_t i = 0; i < hist.size(); ++i) {
		if (norm != 0) {
			hist[i] = hist[i] / norm;
		}
	}
*/
	return hist;

	EXITFUNC;
}





vector<float> EMData::calc_az_dist(int n, float a0, float da, float rmin, float rmax)
{
	ENTERFUNC;

	a0=a0*M_PI/180.0f;
	da=da*M_PI/180.0f;

	if (get_ndim() > 2) {
		throw ImageDimensionException("no 3D image");
	}

	float *yc = new float[n];

	vector<float>	vd(n);
	for (int i = 0; i < n; i++) {
		yc[i] = 0.00001f;
	}

	int isri=is_ri();
	
	float * data = get_data();
	if (is_complex()) {
		int c = 0;
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x += 2, c += 2) {
				int x1 = x / 2;
				int y1 = y<ny/2?y:y-ny;
				float r = (float)Util::hypot_fast(x1,y1);

				if (r >= rmin && r <= rmax) {
					float a = 0;

					if (y != ny / 2 || x != 0) {
						a = (atan2((float)y1, (float)x1) - a0) / da;
					}

					int i = (int)(floor(a));
					a -= i;

					if (i == 0) {
						vd[0] += data[c] * (1.0f - a);
						yc[0] += (1.0f - a);
					}
					else if (i == n - 1) {
						vd[n - 1] += data[c] * a;
						yc[n - 1] += a;
					}
					else if (i > 0 && i < (n - 1)) {
						float h = 0;
						if (isri) {
#ifdef	_WIN32
							h = (float)_hypot(data[c], data[c + 1]);
#else
							h = (float)hypot(data[c], data[c + 1]);
#endif	//_WIN32
						}
						else {
							h = data[c];
						}

						vd[i] += h * (1.0f - a);
						yc[i] += (1.0f - a);
						vd[i + 1] += h * a;
						yc[i + 1] += a;
					}
				}
			}
		}
	}
	else {
		int c = 0;
		float half_nx = (nx - 1) / 2.0f;
		float half_ny = (ny - 1) / 2.0f;

		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++, c++) {
				float y1 = y - half_ny;
				float x1 = x - half_nx;
#ifdef	_WIN32
				float r = (float)_hypot(x1, y1);
#else
				float r = (float)hypot(x1, y1);
#endif

				if (r >= rmin && r <= rmax) {
					float a = 0;
					if (x1 != 0 || y1 != 0) {
						a = atan2(y1, x1);
						if (a < 0) {
							a += M_PI * 2;
						}
					}

					a = (a - a0) / da;
					int i = static_cast < int >(floor(a));
					a -= i;

					if (i == 0) {
						vd[0] += data[c] * (1.0f - a);
						yc[0] += (1.0f - a);
					}
					else if (i == n - 1) {
						vd[n - 1] += data[c] * a;
						yc[n - 1] += (a);
					}
					else if (i > 0 && i < (n - 1)) {
						vd[i] += data[c] * (1.0f - a);
						yc[i] += (1.0f - a);
						vd[i + 1] += data[c] * a;
						yc[i + 1] += a;
					}
				}
			}
		}
	}


	for (int i = 0; i < n; i++) {
		vd[i] /= yc[i];
	}

	if( yc )
	{
		delete[]yc;
		yc = 0;
	}

	return vd;

	EXITFUNC;
}


EMData *EMData::unwrap(int r1, int r2, int xs, int dx, int dy, bool do360, bool weight_radial) const
{
	ENTERFUNC;

	if (get_ndim() != 2) {
		throw ImageDimensionException("2D image only");
	}

	int p = 1;
	if (do360) {
		p = 2;
	}

	if (xs < 1) {
		xs = (int) Util::fast_floor(p * M_PI * ny / 3.0);
		xs-=xs%4;			// 128 bit alignment, though best_fft_size may override
		xs = Util::calc_best_fft_size(xs);
		if (xs<=8) xs=16;
	}

	if (r1 < 0) {
		r1 = 4;
	}

#ifdef	_WIN32
	int rr = ny / 2 - 2 - (int) Util::fast_floor(static_cast<float>(_hypot(dx, dy)));
#else
	int rr = ny / 2 - 2 - (int) Util::fast_floor(static_cast<float>(hypot(dx, dy)));
#endif	//_WIN32
	rr-=rr%2;
	if (r2 <= r1 || r2 > rr) {
		r2 = rr;
	}

	if ( (r2-r1) < 0 ) throw UnexpectedBehaviorException("The combination of function the arguments and the image dimensions causes unexpected behavior internally. Use a larger image, or a smaller value of r1, or a combination of both");

#ifdef EMAN2_USING_CUDA
	if (EMData::usecuda == 1 && isrodataongpu()){
		//cout << " CUDA unwrap" << endl;
		EMData* result = new EMData(0,0,xs,(r2-r1),1);
		if(!result->rw_alloc()) throw UnexpectedBehaviorException("Bad alloc");
		bindcudaarrayA(true);
		emdata_unwrap(result->cudarwdata, r1, r2, xs, p, dx, dy, weight_radial, nx, ny);
		unbindcudaarryA();
		result->update();
		return result;
	}
#endif

	EMData *ret = new EMData();
	ret->set_size(xs, r2 - r1, 1);
	const float *const d = get_const_data();
	float *dd = ret->get_data();
	float pfac = (float)p/(float)xs;
	int nxon2 = nx/2;
	int nyon2 = ny/2;
	for (int x = 0; x < xs; x++) {
		float ang = x * M_PI * pfac;
		float si = sin(ang);
		float co = cos(ang);

		for (int y = 0; y < r2 - r1; y++) {
			float ypr1 = (float)y + r1;
			float xx = ypr1 * co + nxon2 + dx;
			float yy = ypr1 * si + nyon2 + dy;
//			float t = xx - Util::fast_floor(xx);
//			float u = yy - Util::fast_floor(yy);
			float t = xx - (int)xx;
			float u = yy - (int)yy;
//			int k = (int) Util::fast_floor(xx) + (int) (Util::fast_floor(yy)) * nx;
			int k = (int) xx + ((int) yy) * nx;
			float val = Util::bilinear_interpolate(d[k], d[k + 1], d[k + nx], d[k + nx+1], t,u);
			if (weight_radial) val *=  ypr1;
			dd[x + y * xs] = val;
		}

	}
	ret->update();

	EXITFUNC;
	return ret;
}

// NOTE : x axis is from 0 to 0.5  (Nyquist), and thus properly handles non-square images
// complex only
void EMData::apply_radial_func(float x0, float step, vector < float >array, bool interp)
{
	ENTERFUNC;

	if (!is_complex()) throw ImageFormatException("apply_radial_func requires a complex image");

	int n = static_cast < int >(array.size());

	if (n*step>2.0) printf("Warning, apply_radial_func takes x0 and step with respect to Nyquist (0.5)\n");

//	printf("%f %f %f\n",array[0],array[25],array[50]);

	ap2ri();

	size_t ndims = get_ndim();
	float * data = get_data();
	if (ndims == 2) {
		int k = 0;
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i += 2, k += 2) {
				float r;
#ifdef	_WIN32
				if (j<ny/2) r = (float)_hypot(i/(float)(nx*2), j/(float)ny);
				else r = (float)_hypot(i/(float)(nx*2), (ny-j)/(float)ny);
#else
				if (j<ny/2) r = (float)hypot(i/(float)(nx*2), j/(float)ny);
				else r = (float)hypot(i/(float)(nx*2), (ny-j)/(float)ny);
#endif	//_WIN32
				r = (r - x0) / step;

				int l = 0;
				if (interp) {
					l = (int) floor(r);
				}
				else {
					l = (int) floor(r + 1);
				}


				float f = 0;
				if (l >= n - 2) {
					f = array[n - 1];
				}
				else {
					if (interp) {
						r -= l;
						f = (array[l] * (1.0f - r) + array[l + 1] * r);
					}
					else {
						f = array[l];
					}
				}

				data[k] *= f;
				data[k + 1] *= f;
			}
		}
	}
	else if (ndims == 3) {
		int k = 0;
		for (int m = 0; m < nz; m++) {
			float mnz;
			if (m<nz/2) mnz=m*m/(float)(nz*nz);
			else { mnz=(nz-m)/(float)nz; mnz*=mnz; }

			for (int j = 0; j < ny; j++) {
				float jny;
				if (j<ny/2) jny= j*j/(float)(ny*ny);
				else { jny=(ny-j)/(float)ny; jny*=jny; }

				for (int i = 0; i < nx; i += 2, k += 2) {
					float r = std::sqrt((i * i / (nx*nx*4.0f)) + jny + mnz);
					r = (r - x0) / step;

					int l = 0;
					if (interp) {
						l = (int) floor(r);
					}
					else {
						l = (int) floor(r + 1);
					}

					float f = 0;
					if (l >= n - 2) {
						f = array[n - 1];
					}
					else {
						if (interp) {
							r -= l;
							f = (array[l] * (1.0f - r) + array[l + 1] * r);
						}
						else {
							f = array[l];
						}
					}

//if (k%5000==0) printf("%d %d %d   %f\n",i,j,m,f);
					data[k] *= f;
					data[k + 1] *= f;
				}
			}
		}

	}

	update();
	EXITFUNC;
}

vector<float> EMData::calc_radial_dist(int n, float x0, float dx, int inten)
{
	ENTERFUNC;

	vector<double>ret(n);
	vector<double>norm(n);
	vector<double>count(n);

	int x,y,z,i;
	int step=is_complex()?2:1;
	int isinten=get_attr_default("is_intensity",0);
	int isri=is_ri();

	if (isinten&&!inten) { throw InvalidParameterException("Must set inten for calc_radial_dist with intensity image"); }

	switch (inten){
		case 0:
		case 1:
		case 4:
			for (i=0; i<n; i++) ret[i]=norm[i]=count[i]=0.0;
			break;
		case 2:
			for (i=0; i<n; i++) ret[i]=1.0e27;
			break;
		case 3:
			for (i=0; i<n; i++) ret[i]=-1.0e27;
			break;
	}
			
	float * data = get_data();

	// We do 2D separately to avoid the hypot3 call
	if (nz==1) {
		for (y=i=0; y<ny; y++) {
			for (x=0; x<nx; x+=step,i+=step) {
				float r,v;
				int f;
				if (step==2) {		//complex
					if (x==0 && y>ny/2) continue;
					r=(float)(Util::hypot_fast(x/2,y<ny/2?y:ny-y));		// origin at 0,0; periodic
					r=(r-x0)/dx;
					f=int(r);	// safe truncation, so floor isn't needed
					if (f<0 || f>=n) continue;
					switch (inten) {
						case 0:
#ifdef	_WIN32
							if (isri) v=static_cast<float>(_hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#else
							if (isri) v=static_cast<float>(hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#endif
							else v=data[i];							// amp/phase, just get amp
							break;
						case 1:
							if (isinten) v=data[i];
							else if (isri) v=data[i]*data[i]+data[i+1]*data[i+1];
							else v=data[i]*data[i];
							break;
						case 2:
#ifdef	_WIN32
							if (isri) v=static_cast<float>(_hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#else
							if (isri) v=static_cast<float>(hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#endif
							else v=data[i];
							if (v<ret[f]) ret[f]=v;
							break;
						case 3:
#ifdef	_WIN32
							if (isri) v=static_cast<float>(_hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#else
							if (isri) v=static_cast<float>(hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#endif
							else v=data[i];
							if (v>ret[f]) ret[f]=v;
							break;
						case 4:
							if (isinten) v=data[i];
							else if (isri) v=data[i]*data[i]+data[i+1]*data[i+1];
							else v=data[i]*data[i];
							ret[f]+=std::sqrt(v);
							norm[f]+=v;
							count[f]+=1.0;
							break;
					}
				}
				else {
					r=(float)(Util::hypot_fast(x-nx/2,y-ny/2));
					r=(r-x0)/dx;
					f=int(r);	// safe truncation, so floor isn't needed
					if (f<0 || f>=n) continue;
					switch (inten) {
						case 0:
							v=data[i];
							break;
						case 1:
							v=data[i]*data[i];
							break;
						case 2:
							if (data[i]<ret[f]) ret[f]=data[i];
							break;
						case 3:
							if (data[i]>ret[f]) ret[f]=data[i];
							break;
						case 4:
							ret[f]+=data[i];
							norm[f]+=data[i]*data[i];
							count[f]+=1.0;
							break;
					}
				}
				
				if (inten<2) {
					r-=float(f);	// r is now the fractional spacing between bins
	//				printf("%d\t%d\t%d\t%1.3f\t%d\t%1.3f\t%1.4g\n",x,y,f,r,step,Util::hypot_fast(x/2,y<ny/2?y:ny-y),v);
					ret[f]+=v*(1.0f-r);
					norm[f]+=(1.0f-r);
					if (f<n-1) {
						ret[f+1]+=v*r;
						norm[f+1]+=r;
					}
				}
			}
		}
	}
	else {
//		FILE *out = fopen("x.txt","w");
		size_t i;	//3D file may have >2G size
		for (z=i=0; z<nz; ++z) {
			for (y=0; y<ny; ++y) {
				for (x=0; x<nx; x+=step,i+=step) {
					float r,v;
					int f;
					if (step==2) {	//complex
						if (x==0 && z>nz/2) continue;
						if (x==0 && z==nz/2 && y>ny/2) continue;
						r=Util::hypot3(x/2,y<ny/2?y:ny-y,z<nz/2?z:nz-z);	// origin at 0,0; periodic
						r=(r-x0)/dx;
						f=int(r);	// safe truncation, so floor isn't needed
						if (f<0 || f>=n) continue;
						switch(inten) {
							case 0:
#ifdef	_WIN32
								if (isri) v=static_cast<float>(_hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#else
								if (isri) v=static_cast<float>(hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#endif	//_WIN32
								else v=data[i];							// amp/phase, just get amp
								break;
							case 1:
								if (isinten) v=data[i];
								else if (isri) v=data[i]*data[i]+data[i+1]*data[i+1];
								else v=data[i]*data[i];
								break;
							case 2:
#ifdef	_WIN32
								if (isri) v=static_cast<float>(_hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#else
								if (isri) v=static_cast<float>(hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#endif
								else v=data[i];							// amp/phase, just get amp
								if (v<ret[f]) ret[f]=v;
								break;
							case 3:
#ifdef	_WIN32
								if (isri) v=static_cast<float>(_hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#else
								if (isri) v=static_cast<float>(hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#endif
								else v=data[i];							// amp/phase, just get amp
								if (v>ret[f]) ret[f]=v;
								break;
							case 4:
								if (isinten) v=data[i];
								else if (isri) v=data[i]*data[i]+data[i+1]*data[i+1];
								else v=data[i]*data[i];
								ret[f]+=std::sqrt(v);
								norm[f]+=v;
								count[f]+=1.0;
//								if (f==100) fprintf(out,"%1.4g\n",std::sqrt(v));
								break;
						}						
					}
					else {
						r=Util::hypot3(x-nx/2,y-ny/2,z-nz/2);
						r=(r-x0)/dx;
						f=int(r);	// safe truncation, so floor isn't needed
						if (f<0 || f>=n) continue;
						switch(inten) {
							case 0:
								v=data[i];
								break;
							case 1:
								v=data[i]*data[i];
								break;
							case 2:
								if (data[i]<ret[f]) ret[f]=data[i];
								break;
							case 3:
								if (data[i]>ret[f]) ret[f]=data[i];
								break;
							case 4:
								ret[f]+=data[i];
								norm[f]+=data[i]*data[i];
								count[f]+=1.0;
								break;
						}
					}

					if (inten<2) {
// 						ret[f]+=v;
// 						norm[f]+=1.0;
						
						r-=float(f);	// r is now the fractional spacing between bins
						ret[f]+=v*(1.0f-r);
						norm[f]+=(1.0f-r);
						if (f<n-1) {
							ret[f+1]+=v*r;
							norm[f+1]+=r;
						}
					}
				}
			}
		}
//		fclose(out);
	}
	
	if (inten<2) {
		for (i=0; i<n; i++) ret[i]/=(norm[i]==0?1.0f:norm[i]);	// Normalize
	}
	else if (inten==4) {
		for (i=0; i<n; i++) {
			ret[i]/=count[i];	// becomes mean
			norm[i]/=count[i];	// avg amp^2
			ret[i]=std::sqrt(norm[i]-ret[i]*ret[i]);	// sigma
		}
	}
		
	EXITFUNC;

	return vector<float>(ret.begin(),ret.end());
}

vector<float> EMData::calc_radial_dist(int n, float x0, float dx, int nwedge, float offset, bool inten)
{
	ENTERFUNC;

	if (nz > 1) {
		LOGERR("2D images only.");
		throw ImageDimensionException("2D images only");
	}
	int isinten=get_attr_default("is_intensity",0);

	if (isinten&&!inten) { throw InvalidParameterException("Must set inten for calc_radial_dist with intensity image"); }


	vector<double>ret(n*nwedge);
	vector<double>norm(n*nwedge);

	int x,y,i;
	int isri = is_ri();	// this has become expensive!
	int step=is_complex()?2:1;
	float astep=static_cast<float>(M_PI*2.0/nwedge);
	if (is_complex()) astep/=2;							// Since we only have the right 1/2 of Fourier space
	float* data = get_data();
	for (i=0; i<n*nwedge; i++) ret[i]=norm[i]=0.0;

	// We do 2D separately to avoid the hypot3 call
	for (y=i=0; y<ny; y++) {
		for (x=0; x<nx; x+=step,i+=step) {
			float r,v,a;
			int bin;
			if (is_complex()) {
#ifdef	_WIN32
				r=static_cast<float>(_hypot(x/2.0,y<ny/2?y:ny-y));		// origin at 0,0; periodic
#else
				r=static_cast<float>(hypot(x/2.0,y<ny/2?y:ny-y));		// origin at 0,0; periodic
#endif
				a=atan2(float(y<ny/2?y:y-ny),x/2.0f);
				if (!inten) {
#ifdef	_WIN32
					if (isri) v=static_cast<float>(_hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#else
					if (isri) v=static_cast<float>(hypot(data[i],data[i+1]));	// real/imag, compute amplitude
#endif	//_WIN32
					else v=data[i];							// amp/phase, just get amp
				} else {
					if (isinten) v=data[i];
					else if (isri) v=data[i]*data[i]+data[i+1]*data[i+1];
					else v=data[i]*data[i];
				}
				bin=n*int(floor((a+M_PI/2.0f+offset)/astep));
			}
			else {
#ifdef	_WIN32
				r=static_cast<float>(_hypot(x-nx/2,y-ny/2));
#else
				r=static_cast<float>(hypot(x-nx/2,y-ny/2));
#endif	//_WIN32
				a=atan2(float(y-ny/2),float(x-nx/2));
				if (inten) v=data[i]*data[i];
				else v=data[i];
				bin=n*int(floor((a+M_PI+offset)/astep));
			}
			if (bin>=nwedge*n) bin-=nwedge*n;
			if (bin<0) bin+=nwedge*n;
			r=(r-x0)/dx;
			int f=int(r);	// safe truncation, so floor isn't needed
			r-=float(f);	// r is now the fractional spacing between bins
//			printf("%d %d %d %d %1.3f %1.3f\n",x,y,bin,f,r,a);
			if (f>=0 && f<n) {
				ret[f+bin]+=v*(1.0f-r);
				norm[f+bin]+=(1.0f-r);
				if (f<n-1) {
					ret[f+1+bin]+=v*r;
					norm[f+1+bin]+=r;
				}
			}
		}
	}

	for (i=0; i<n*nwedge; i++) ret[i]/=norm[i]?norm[i]:1.0f;	// Normalize
	EXITFUNC;

	return vector<float>(ret.begin(),ret.end());
}

void EMData::cconj() {
	ENTERFUNC;
	if (!is_complex() || !is_ri())
		throw ImageFormatException("EMData::conj requires a complex, ri image");
	int nxreal = nx -2 + int(is_fftodd());
	int nxhalf = nxreal/2;
	for (int iz = 0; iz < nz; iz++)
		for (int iy = 0; iy < ny; iy++)
			for (int ix = 0; ix <= nxhalf; ix++)
				cmplx(ix,iy,iz) = conj(cmplx(ix,iy,iz));
	EXITFUNC;
}

void EMData::update_stat() const
{
	ENTERFUNC;
//	printf("update stat %f %d\n",(float)attr_dict["mean"],flags);
	if (!(flags & EMDATA_NEEDUPD))
	{
		EXITFUNC;
		return;
	}
	if (rdata==0) return;

	float* data = get_data();
	float max = -FLT_MAX;
	float min = -max;

	double sum = 0;
	double square_sum = 0;

	int step = 1;
	if (is_complex() && !is_ri()) {
		step = 2;
	}

	int n_nonzero = 0;

	size_t size = (size_t)nx*ny*nz;
	for (size_t i = 0; i < size; i += step) {
		float v = data[i];
	#ifdef _WIN32
		max = _cpp_max(max,v);
		min = _cpp_min(min,v);
	#else
		max=std::max<float>(max,v);
		min=std::min<float>(min,v);
	#endif	//_WIN32
		sum += v;
		square_sum += v * (double)(v);
		if (v != 0) n_nonzero++;
	}

	size_t n = size / step;
	double mean = sum / n;

#ifdef _WIN32
	float sigma = (float)std::sqrt( _cpp_max(0.0,(square_sum - sum*sum / n)/(n-1)));
	n_nonzero = _cpp_max(1,n_nonzero);
	double sigma_nonzero = std::sqrt( _cpp_max(0,(square_sum  - sum*sum/n_nonzero)/(n_nonzero-1)));
#else
	float sigma = (float)std::sqrt(std::max<double>(0.0,(square_sum - sum*sum / n)/(n-1)));
	n_nonzero = std::max<int>(1,n_nonzero);
	double sigma_nonzero = std::sqrt(std::max<double>(0,(square_sum  - sum*sum/n_nonzero)/(n_nonzero-1)));
#endif	//_WIN32
	double mean_nonzero = sum / n_nonzero; // previous version overcounted! G2

	attr_dict["minimum"] = min;
	attr_dict["maximum"] = max;
	attr_dict["mean"] = (float)(mean);
	attr_dict["sigma"] = (float)(sigma);
	attr_dict["square_sum"] = (float)(square_sum);
	attr_dict["mean_nonzero"] = (float)(mean_nonzero);
	attr_dict["sigma_nonzero"] = (float)(sigma_nonzero);
	attr_dict["is_complex"] = (int) is_complex();
	attr_dict["is_complex_ri"] = (int) is_ri();

	flags &= ~EMDATA_NEEDUPD;

	if (rot_fp != 0)
	{
		delete rot_fp; rot_fp = 0;
	}

	EXITFUNC;
//	printf("done stat %f %f %f\n",(float)mean,(float)max,(float)sigma);
}

/**
 * Change the equality check for memory address check, i.e. two EMData objects are considered equal
 * only when they are same object from the same memory address.
 */
bool EMData::operator==(const EMData& that) const {
	if(this != &that) {
		return false;
	}
	else {
		return true;
	}
}

bool EMData::equal(const EMData& that) const {
	if (that.get_xsize() != nx || that.get_ysize() != ny || that.get_zsize() != nz ) return false;

	const float*  d1 = that.get_const_data();
	float* d2 = get_data();

	for(size_t i =0; i < get_size(); ++i,++d1,++d2) {
		if ((*d1) != (*d2)) return false;
	}

//	if(attr_dict != that.attr_dict) {
//		return false;
//	}

	return true;
}

EMData * EMAN::operator+(const EMData & em, float n)
{
	EMData * r = em.copy();
	r->add(n);
	return r;
}

EMData * EMAN::operator-(const EMData & em, float n)
{
	EMData* r = em.copy();
	r->sub(n);
	return r;
}

EMData * EMAN::operator*(const EMData & em, float n)
{
	EMData* r = em.copy();
	r ->mult(n);
	return r;
}

EMData * EMAN::operator/(const EMData & em, float n)
{
	EMData * r = em.copy();
	r->div(n);
	return r;
}


EMData * EMAN::operator+(float n, const EMData & em)
{
	EMData * r = em.copy();
	r->add(n);
	return r;
}

EMData * EMAN::operator-(float n, const EMData & em)
{
	EMData * r = em.copy();
	r->mult(-1.0f);
	r->add(n);
	return r;
}

EMData * EMAN::operator*(float n, const EMData & em)
{
	EMData * r = em.copy();
	r->mult(n);
	return r;
}

EMData * EMAN::operator/(float n, const EMData & em)
{
	EMData * r = em.copy();
	r->to_one();
	r->mult(n);
	r->div(em);

	return r;
}

EMData * EMAN::rsub(const EMData & em, float n)
{
	return EMAN::operator-(n, em);
}

EMData * EMAN::rdiv(const EMData & em, float n)
{
	return EMAN::operator/(n, em);
}

EMData * EMAN::operator+(const EMData & a, const EMData & b)
{
	EMData * r = a.copy();
	r->add(b);
	return r;
}

EMData * EMAN::operator-(const EMData & a, const EMData & b)
{
	EMData * r = a.copy();
	r->sub(b);
	return r;
}

EMData * EMAN::operator*(const EMData & a, const EMData & b)
{
	EMData * r = a.copy();
	r->mult(b);
	return r;
}

EMData * EMAN::operator/(const EMData & a, const EMData & b)
{
	EMData * r = a.copy();
	r->div(b);
	return r;
}

void EMData::set_xyz_origin(float origin_x, float origin_y, float origin_z)
{
	attr_dict["origin_x"] = origin_x;
	attr_dict["origin_y"] = origin_y;
	attr_dict["origin_z"] = origin_z;
}

#if 0
void EMData::calc_rcf(EMData * with, vector < float >&sum_array)
{
	ENTERFUNC;

	int array_size = sum_array.size();
	float da = 2 * M_PI / array_size;
	float *dat = new float[array_size + 2];
	float *dat2 = new float[array_size + 2];
	int nx2 = nx * 9 / 20;

	float lim = 0;
	if (fabs(translation[0]) < fabs(translation[1])) {
		lim = fabs(translation[1]);
	}
	else {
		lim = fabs(translation[0]);
	}

	nx2 -= static_cast < int >(floor(lim));

	for (int i = 0; i < array_size; i++) {
		sum_array[i] = 0;
	}

	float sigma = attr_dict["sigma"];
	float with_sigma = with->get_attr_dict().get("sigma");

	vector<float> vdata, vdata2;
	for (int i = 8; i < nx2; i += 6) {
		vdata = calc_az_dist(array_size, 0, da, i, i + 6);
		vdata2 = with->calc_az_dist(array_size, 0, da, i, i + 6);
		Assert(vdata.size() <= array_size + 2);
		Assert(cdata2.size() <= array_size + 2);
		std::copy(vdata.begin(), vdata.end(), dat);
		std::copy(vdata2.begin(), vdata2.end(), dat2);

		EMfft::real_to_complex_1d(dat, dat, array_size);
		EMfft::real_to_complex_1d(dat2, dat2, array_size);

		for (int j = 0; j < array_size + 2; j += 2) {
			float max = dat[j] * dat2[j] + dat[j + 1] * dat2[j + 1];
			float max2 = dat[j + 1] * dat2[j] - dat2[j + 1] * dat[j];
			dat[j] = max;
			dat[j + 1] = max2;
		}

		EMfft::complex_to_real_1d(dat, dat, array_size);
		float norm = array_size * array_size * (4.0f * sigma) * (4.0f * with_sigma);

		for (int j = 0; j < array_size; j++) {
			sum_array[j] += dat[j] * (float) i / norm;
		}
	}

	if( dat )
	{
		delete[]dat;
		dat = 0;
	}

	if( dat2 )
	{
		delete[]dat2;
		dat2 = 0;
	}
	EXITFUNC;
}

#endif

void EMData::add_incoherent(EMData * obj)
{
	ENTERFUNC;

	if (!obj) {
		LOGERR("NULL image");
		throw NullPointerException("NULL image");
	}

	if (!obj->is_complex() || !is_complex()) {
		throw ImageFormatException("complex images only");
	}

	if (!EMUtil::is_same_size(this, obj)) {
		throw ImageFormatException("images not same size");
	}

	ri2ap();
	obj->ri2ap();

	float *dest = get_data();
	float *src = obj->get_data();
	size_t size = (size_t)nx * ny * nz;
	for (size_t j = 0; j < size; j += 2) {
#ifdef	_WIN32
		dest[j] = (float) _hypot(src[j], dest[j]);
#else
		dest[j] = (float) hypot(src[j], dest[j]);
#endif	//_WIN32
		dest[j + 1] = 0;
	}

	obj->update();
	update();
	EXITFUNC;
}


float EMData::calc_dist(EMData * second_img, int y_index) const
{
	ENTERFUNC;

	if (get_ndim() != 1) {
		throw ImageDimensionException("'this' image is 1D only");
	}

	if (second_img->get_xsize() != nx || ny != 1) {
		throw ImageFormatException("image xsize not same");
	}

	if (y_index > second_img->get_ysize() || y_index < 0) {
		return -1;
	}

	float ret = 0;
	float *d1 = get_data();
	float *d2 = second_img->get_data() + second_img->get_xsize() * y_index;

	for (int i = 0; i < nx; i++) {
		ret += Util::square(d1[i] - d2[i]);
	}
	EXITFUNC;
	return std::sqrt(ret);
}


EMData * EMData::calc_fast_sigma_image( EMData* mask)
{
	ENTERFUNC;

	bool maskflag = false;
	if (mask == 0) {
		mask = new EMData(nx,ny,nz);
		mask->process_inplace("testimage.circlesphere");
		maskflag = true;
	}

	if (get_ndim() != mask->get_ndim() ) throw ImageDimensionException("The dimensions do not match");

	int mnx = mask->get_xsize(); int mny = mask->get_ysize(); int mnz = mask->get_zsize();

	if ( mnx > nx || mny > ny || mnz > nz)
		throw ImageDimensionException("Can not calculate variance map using an image that is larger than this image");

	size_t P = 0;
	for(size_t i = 0; i < mask->get_size(); ++i){
		if (mask->get_value_at(i) != 0){
			++P;
		}
	}
	float normfac = 1.0f/(float)P;

//	bool undoclip = false;

	int nxc = nx+mnx; int nyc = ny+mny; int nzc = nz+mnz;
//	if ( mnx < nx || mny < ny || mnz < nz) {
	Region r;
	if (ny == 1) r = Region((mnx-nxc)/2,nxc);
	else if (nz == 1) r = Region((mnx-nxc)/2, (mny-nyc)/2,nxc,nyc);
	else r = Region((mnx-nxc)/2, (mny-nyc)/2,(mnz-nzc)/2,nxc,nyc,nzc);
	mask->clip_inplace(r,0.0);
	//Region r((mnx-nxc)/2, (mny-nyc)/2,(mnz-nzc)/2,nxc,nyc,nzc);
	//mask->clip_inplace(r);
	//undoclip = true;
	//}

	// Here we generate the local average of the squares
	Region r2;
	if (ny == 1) r2 = Region((nx-nxc)/2,nxc);
	else if (nz == 1) r2 = Region((nx-nxc)/2, (ny-nyc)/2,nxc,nyc);
	else r2 = Region((nx-nxc)/2, (ny-nyc)/2,(nz-nzc)/2,nxc,nyc,nzc);
	EMData* squared = get_clip(r2,get_edge_mean());

	EMData* tmp = squared->copy();
	Dict pow;
	pow["pow"] = 2.0f;
	squared->process_inplace("math.pow",pow);
	EMData* s = mask->convolute(squared);//ming, mask squared exchange
	squared->mult(normfac);

	EMData* m = mask->convolute(tmp);//ming, tmp mask exchange
	m->mult(normfac);
	m->process_inplace("math.pow",pow);
	delete tmp; tmp = 0;
	s->sub(*m);
	// Here we finally generate the standard deviation image
	s->process_inplace("math.sqrt");

//	if ( undoclip ) {
//		Region r((nx-mnx)/2, (ny-mny)/2, (nz-mnz)/2,mnx,mny,mnz);
//		mask->clip_inplace(r);
//	}

	if (maskflag) {
		delete mask;
		mask = 0;
	} else {
		Region r;
		if (ny == 1) r = Region((nxc-mnx)/2,mnx);
		else if (nz == 1) r = Region((nxc-mnx)/2, (nyc-mny)/2,mnx,mny);
		else r = Region((nxc-mnx)/2, (nyc-mny)/2,(nzc-mnz)/2,mnx,mny,mnz);
		mask->clip_inplace(r);
	}

	delete squared;
	delete m;

	s->process_inplace("xform.phaseorigin.tocenter");
	Region r3;
	if (ny == 1) r3 = Region((nxc-nx)/2,nx);
	else if (nz == 1) r3 = Region((nxc-nx)/2, (nyc-ny)/2,nx,ny);
	else r3 = Region((nxc-nx)/2, (nyc-ny)/2,(nzc-nz)/2,nx,ny,nz);
	s->clip_inplace(r3);
	EXITFUNC;
	return s;
}

//  The following code looks strange - does anybody know it?  Please let me know, pawel.a.penczek@uth.tmc.edu  04/09/06.
// This is just an implementation of "Roseman's" fast normalized cross-correlation (Ultramicroscopy, 2003). But the contents of this function have changed dramatically since you wrote that comment (d.woolford).
EMData *EMData::calc_flcf(EMData * with)
{
	ENTERFUNC;
	EMData *this_copy=this;
	this_copy=copy();

	int mnx = with->get_xsize(); int mny = with->get_ysize(); int mnz = with->get_zsize();
	int nxc = nx+mnx; int nyc = ny+mny; int nzc = nz+mnz;

	// Ones is a circular/spherical mask, consisting of 1s.
	EMData* ones = new EMData(mnx,mny,mnz);
	ones->process_inplace("testimage.circlesphere");

	// Get a copy of with, we will eventually resize it
	EMData* with_resized = with->copy();
	with_resized->process_inplace("normalize");
	with_resized->mult(*ones);

	EMData* s = calc_fast_sigma_image(ones);// Get the local sigma image

	Region r1;
	if (ny == 1) r1 = Region((mnx-nxc)/2,nxc);
	else if (nz == 1) r1 = Region((mnx-nxc)/2, (mny-nyc)/2,nxc,nyc);
	else r1 = Region((mnx-nxc)/2, (mny-nyc)/2,(mnz-nzc)/2,nxc,nyc,nzc);
	with_resized->clip_inplace(r1,0.0);

	Region r2;
	if (ny == 1) r2 = Region((nx-nxc)/2,nxc);
	else if (nz == 1) r2 = Region((nx-nxc)/2, (ny-nyc)/2,nxc,nyc);
	else r2 = Region((nx-nxc)/2, (ny-nyc)/2,(nz-nzc)/2,nxc,nyc,nzc);
	this_copy->clip_inplace(r2,0.0);

	EMData* corr = this_copy->calc_ccf(with_resized); // the ccf results should have same size as sigma

	corr->process_inplace("xform.phaseorigin.tocenter");
	Region r3;
	if (ny == 1) r3 = Region((nxc-nx)/2,nx);
	else if (nz == 1) r3 = Region((nxc-nx)/2, (nyc-ny)/2,nx,ny);
	else r3 = Region((nxc-nx)/2, (nyc-ny)/2,(nzc-nz)/2,nx,ny,nz);
	corr->clip_inplace(r3);

	corr->div(*s);

	delete with_resized; delete ones; delete this_copy; delete s;
	EXITFUNC;
	return corr;
}

EMData *EMData::convolute(EMData * with)
{
	ENTERFUNC;

	EMData *f1 = do_fft();
	if (!f1) {
		LOGERR("FFT returns NULL image");
		throw NullPointerException("FFT returns NULL image");
	}

	f1->ap2ri();

	EMData *cf = 0;
	if (with) {
		cf = with->do_fft();
		if (!cf) {
			LOGERR("FFT returns NULL image");
			throw NullPointerException("FFT returns NULL image");
		}
		cf->ap2ri();
	}
	else {
		cf = f1->copy();
	}
	//printf("cf_x=%d, f1y=%d, thisx=%d, withx=%d\n",cf->get_xsize(),f1->get_ysize(),this->get_xsize(),with->get_xsize());
	if (with && !EMUtil::is_same_size(f1, cf)) {
		LOGERR("images not same size");
		throw ImageFormatException("images not same size");
	}

	float *rdata1 = f1->get_data();
	float *rdata2 = cf->get_data();
	size_t cf_size = (size_t)cf->get_xsize() * cf->get_ysize() * cf->get_zsize();

	float re,im;

	for (size_t i = 0; i < cf_size; i += 2) {
		re = rdata1[i] * rdata2[i] - rdata1[i + 1] * rdata2[i + 1];
		im = rdata1[i + 1] * rdata2[i] + rdata1[i] * rdata2[i + 1];
		rdata2[i]=re;
		rdata2[i+1]=im;
	}
	cf->update();
	EMData *f2 = cf->do_ift();//ming change cf to cf_temp
	//printf("cf_x=%d, f2x=%d, thisx=%d, withx=%d\n",cf->get_xsize(),f2->get_xsize(),this->get_xsize(),with->get_xsize());
	if( cf )
	{
		delete cf;
		cf = 0;
	}

	if( f1 )
	{
		delete f1;
		f1=0;
	}

	EXITFUNC;
	return f2;
}


void EMData::common_lines(EMData * image1, EMData * image2,
						  int mode, int steps, bool horizontal)
{
	ENTERFUNC;

	if (!image1 || !image2) {
		throw NullPointerException("NULL image");
	}

	if (mode < 0 || mode > 2) {
		throw OutofRangeException(0, 2, mode, "invalid mode");
	}

	if (!image1->is_complex()) {
		image1 = image1->do_fft();
	}
	if (!image2->is_complex()) {
		image2 = image2->do_fft();
	}

	image1->ap2ri();
	image2->ap2ri();

	if (!EMUtil::is_same_size(image1, image2)) {
		throw ImageFormatException("images not same sizes");
	}

	int image2_nx = image2->get_xsize();
	int image2_ny = image2->get_ysize();

	int rmax = image2_ny / 4 - 1;
	int array_size = steps * rmax * 2;
	float *im1 = new float[array_size];
	float *im2 = new float[array_size];
	for (int i = 0; i < array_size; i++) {
		im1[i] = 0;
		im2[i] = 0;
	}

	set_size(steps * 2, steps * 2, 1);

	float *image1_data = image1->get_data();
	float *image2_data = image2->get_data();

	float da = M_PI / steps;
	float a = -M_PI / 2.0f + da / 2.0f;
	int jmax = 0;

	for (int i = 0; i < steps * 2; i += 2, a += da) {
		float s1 = 0;
		float s2 = 0;
		int i2 = i * rmax;
		int j = 0;

		for (float r = 3.0f; r < rmax - 3.0f; j += 2, r += 1.0f) {
			float x = r * cos(a);
			float y = r * sin(a);

			if (x < 0) {
				x = -x;
				y = -y;
				LOGERR("CCL ERROR %d, %f !\n", i, -x);
			}

			int k = (int) (floor(x) * 2 + floor(y + image2_ny / 2) * image2_nx);
			int l = i2 + j;
			float x2 = x - floor(x);
			float y2 = y - floor(y);

			im1[l] = Util::bilinear_interpolate(image1_data[k],
												image1_data[k + 2],
												image1_data[k + image2_nx],
												image1_data[k + 2 + image2_nx], x2, y2);

			im2[l] = Util::bilinear_interpolate(image2_data[k],
												image2_data[k + 2],
												image2_data[k + image2_nx],
												image2_data[k + 2 + image2_nx], x2, y2);

			k++;

			im1[l + 1] = Util::bilinear_interpolate(image1_data[k],
													image1_data[k + 2],
													image1_data[k + image2_nx],
													image1_data[k + 2 + image2_nx], x2, y2);

			im2[l + 1] = Util::bilinear_interpolate(image2_data[k],
													image2_data[k + 2],
													image2_data[k + image2_nx],
													image2_data[k + 2 + image2_nx], x2, y2);

			s1 += Util::square_sum(im1[l], im1[l + 1]);
			s2 += Util::square_sum(im2[l], im2[l + 1]);
		}

		jmax = j - 1;
		float sqrt_s1 = std::sqrt(s1);
		float sqrt_s2 = std::sqrt(s2);

		int l = 0;
		for (float r = 1; r < rmax; r += 1.0f) {
			int i3 = i2 + l;
			im1[i3] /= sqrt_s1;
			im1[i3 + 1] /= sqrt_s1;
			im2[i3] /= sqrt_s2;
			im2[i3 + 1] /= sqrt_s2;
			l += 2;
		}
	}
	float * data = get_data();

	switch (mode) {
	case 0:
		for (int m1 = 0; m1 < 2; m1++) {
			for (int m2 = 0; m2 < 2; m2++) {

				if (m1 == 0 && m2 == 0) {
					for (int i = 0; i < steps; i++) {
						int i2 = i * rmax * 2;
						for (int j = 0; j < steps; j++) {
							int l = i + j * steps * 2;
							int j2 = j * rmax * 2;
							data[l] = 0;
							for (int k = 0; k < jmax; k++) {
								data[l] += im1[i2 + k] * im2[j2 + k];
							}
						}
					}
				}
				else {
					int steps2 = steps * m2 + steps * steps * 2 * m1;

					for (int i = 0; i < steps; i++) {
						int i2 = i * rmax * 2;
						for (int j = 0; j < steps; j++) {
							int j2 = j * rmax * 2;
							int l = i + j * steps * 2 + steps2;
							data[l] = 0;

							for (int k = 0; k < jmax; k += 2) {
								i2 += k;
								j2 += k;
								data[l] += im1[i2] * im2[j2];
								data[l] += -im1[i2 + 1] * im2[j2 + 1];
							}
						}
					}
				}
			}
		}

		break;
	case 1:
		for (int m1 = 0; m1 < 2; m1++) {
			for (int m2 = 0; m2 < 2; m2++) {
				int steps2 = steps * m2 + steps * steps * 2 * m1;
				int p1_sign = 1;
				if (m1 != m2) {
					p1_sign = -1;
				}

				for (int i = 0; i < steps; i++) {
					int i2 = i * rmax * 2;

					for (int j = 0; j < steps; j++) {
						int j2 = j * rmax * 2;

						int l = i + j * steps * 2 + steps2;
						data[l] = 0;
						float a = 0;

						for (int k = 0; k < jmax; k += 2) {
							i2 += k;
							j2 += k;

#ifdef	_WIN32
							float a1 = (float) _hypot(im1[i2], im1[i2 + 1]);
#else
							float a1 = (float) hypot(im1[i2], im1[i2 + 1]);
#endif	//_WIN32
							float p1 = atan2(im1[i2 + 1], im1[i2]);
							float p2 = atan2(im2[j2 + 1], im2[j2]);

							data[l] += Util::angle_sub_2pi(p1_sign * p1, p2) * a1;
							a += a1;
						}

						data[l] /= (float)(a * M_PI / 180.0f);
					}
				}
			}
		}

		break;
	case 2:
		for (int m1 = 0; m1 < 2; m1++) {
			for (int m2 = 0; m2 < 2; m2++) {
				int steps2 = steps * m2 + steps * steps * 2 * m1;

				for (int i = 0; i < steps; i++) {
					int i2 = i * rmax * 2;

					for (int j = 0; j < steps; j++) {
						int j2 = j * rmax * 2;
						int l = i + j * steps * 2 + steps2;
						data[l] = 0;

						for (int k = 0; k < jmax; k += 2) {
							i2 += k;
							j2 += k;
#ifdef	_WIN32
							data[l] += (float) (_hypot(im1[i2], im1[i2 + 1]) * _hypot(im2[j2], im2[j2 + 1]));
#else
							data[l] += (float) (hypot(im1[i2], im1[i2 + 1]) * hypot(im2[j2], im2[j2 + 1]));
#endif	//_WIN32
						}
					}
				}
			}
		}

		break;
	default:
		break;
	}

	if (horizontal) {
		float *tmp_array = new float[ny];
		for (int i = 1; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				tmp_array[j] = get_value_at(i, j);
			}
			for (int j = 0; j < ny; j++) {
				set_value_at(i, j, 0, tmp_array[(j + i) % ny]);
			}
		}
		if( tmp_array )
		{
			delete[]tmp_array;
			tmp_array = 0;
		}
	}

	if( im1 )
	{
		delete[]im1;
		im1 = 0;
	}

	if( im2 )
	{
		delete im2;
		im2 = 0;
	}


	image1->update();
	image2->update();
	if( image1 )
	{
		delete image1;
		image1 = 0;
	}
	if( image2 )
	{
		delete image2;
		image2 = 0;
	}
	update();
	EXITFUNC;
}



void EMData::common_lines_real(EMData * image1, EMData * image2,
							   int steps, bool horiz)
{
	ENTERFUNC;

	if (!image1 || !image2) {
		throw NullPointerException("NULL image");
	}

	if (!EMUtil::is_same_size(image1, image2)) {
		throw ImageFormatException("images not same size");
	}

	int steps2 = steps * 2;
	int image_ny = image1->get_ysize();
	EMData *image1_copy = image1->copy();
	EMData *image2_copy = image2->copy();

	float *im1 = new float[steps2 * image_ny];
	float *im2 = new float[steps2 * image_ny];

	EMData *images[] = { image1_copy, image2_copy };
	float *ims[] = { im1, im2 };

	for (int m = 0; m < 2; m++) {
		float *im = ims[m];
		float a = M_PI / steps2;
		Transform t(Dict("type","2d","alpha",-a));
		for (int i = 0; i < steps2; i++) {
			images[i]->transform(t);
			float *data = images[i]->get_data();

			for (int j = 0; j < image_ny; j++) {
				float sum = 0;
				for (int k = 0; k < image_ny; k++) {
					sum += data[j * image_ny + k];
				}
				im[i * image_ny + j] = sum;
			}

			float sum1 = 0;
			float sum2 = 0;
			for (int j = 0; j < image_ny; j++) {
				int l = i * image_ny + j;
				sum1 += im[l];
				sum2 += im[l] * im[l];
			}

			float mean = sum1 / image_ny;
			float sigma = std::sqrt(sum2 / image_ny - sum1 * sum1);

			for (int j = 0; j < image_ny; j++) {
				int l = i * image_ny + j;
				im[l] = (im[l] - mean) / sigma;
			}

			images[i]->update();
			a += M_PI / steps;
		}
	}

	set_size(steps2, steps2, 1);
	float *data1 = get_data();

	if (horiz) {
		for (int i = 0; i < steps2; i++) {
			for (int j = 0, l = i; j < steps2; j++, l++) {
				if (l == steps2) {
					l = 0;
				}

				float sum = 0;
				for (int k = 0; k < image_ny; k++) {
					sum += im1[i * image_ny + k] * im2[l * image_ny + k];
				}
				data1[i + j * steps2] = sum;
			}
		}
	}
	else {
		for (int i = 0; i < steps2; i++) {
			for (int j = 0; j < steps2; j++) {
				float sum = 0;
				for (int k = 0; k < image_ny; k++) {
					sum += im1[i * image_ny + k] * im2[j * image_ny + k];
				}
				data1[i + j * steps2] = sum;
			}
		}
	}

	update();

	if( image1_copy )
	{
		delete image1_copy;
		image1_copy = 0;
	}

	if( image2_copy )
	{
		delete image2_copy;
		image2_copy = 0;
	}

	if( im1 )
	{
		delete[]im1;
		im1 = 0;
	}

	if( im2 )
	{
		delete[]im2;
		im2 = 0;
	}
	EXITFUNC;
}


void EMData::cut_slice(const EMData *const map, const Transform& transform, bool interpolate)
{
	ENTERFUNC;

	if (!map) throw NullPointerException("NULL image");
	// These restrictions should be ultimately restricted so that all that matters is get_ndim() = (map->get_ndim() -1)
	if ( get_ndim() != 2 ) throw ImageDimensionException("Can not call cut slice on an image that is not 2D");
	if ( map->get_ndim() != 3 ) throw ImageDimensionException("Can not cut slice from an image that is not 3D");
	// Now check for complex images - this is really just being thorough
	if ( is_complex() ) throw ImageFormatException("Can not call cut slice on an image that is complex");
	if ( map->is_complex() ) throw ImageFormatException("Can not cut slice from a complex image");


	float *sdata = map->get_data();
	float *ddata = get_data();

	int map_nx = map->get_xsize();
	int map_ny = map->get_ysize();
	int map_nz = map->get_zsize();
	int map_nxy = map_nx * map_ny;

	int ymax = ny/2;
	if ( ny % 2 == 1 ) ymax += 1;
	int xmax = nx/2;
	if ( nx % 2 == 1 ) xmax += 1;
	for (int y = -ny/2; y < ymax; y++) {
		for (int x = -nx/2; x < xmax; x++) {
			Vec3f coord(x,y,0);
			Vec3f soln = transform*coord;

// 			float xx = (x+pretrans[0]) * (*ort)[0][0] +  (y+pretrans[1]) * (*ort)[0][1] + pretrans[2] * (*ort)[0][2] + posttrans[0];
// 			float yy = (x+pretrans[0]) * (*ort)[1][0] +  (y+pretrans[1]) * (*ort)[1][1] + pretrans[2] * (*ort)[1][2] + posttrans[1];
// 			float zz = (x+pretrans[0]) * (*ort)[2][0] +  (y+pretrans[1]) * (*ort)[2][1] + pretrans[2] * (*ort)[2][2] + posttrans[2];


// 			xx += map_nx/2;
// 			yy += map_ny/2;
// 			zz += map_nz/2;

			float xx = soln[0]+map_nx/2;
			float yy = soln[1]+map_ny/2;
			float zz = soln[2]+map_nz/2;

			int l = (x+nx/2) + (y+ny/2) * nx;

			float t = xx - floor(xx);
			float u = yy - floor(yy);
			float v = zz - floor(zz);

			if (xx < 0 || yy < 0 || zz < 0 ) {
				ddata[l] = 0;
				continue;
			}
			if (interpolate) {
				if ( xx > map_nx - 1 || yy > map_ny - 1 || zz > map_nz - 1) {
					ddata[l] = 0;
					continue;
				}
				int k = (int) (Util::fast_floor(xx) + Util::fast_floor(yy) * map_nx + Util::fast_floor(zz) * map_nxy);


				if (xx < (map_nx - 1) && yy < (map_ny - 1) && zz < (map_nz - 1)) {
					ddata[l] = Util::trilinear_interpolate(sdata[k],
								sdata[k + 1], sdata[k + map_nx],sdata[k + map_nx + 1],
								sdata[k + map_nxy], sdata[k + map_nxy + 1], sdata[k + map_nx + map_nxy],
								sdata[k + map_nx + map_nxy + 1],t, u, v);
				}
				else if ( xx == (map_nx - 1) && yy == (map_ny - 1) && zz == (map_nz - 1) ) {
					ddata[l] += sdata[k];
				}
				else if ( xx == (map_nx - 1) && yy == (map_ny - 1) ) {
					ddata[l] +=	Util::linear_interpolate(sdata[k], sdata[k + map_nxy],v);
				}
				else if ( xx == (map_nx - 1) && zz == (map_nz - 1) ) {
					ddata[l] += Util::linear_interpolate(sdata[k], sdata[k + map_nx],u);
				}
				else if ( yy == (map_ny - 1) && zz == (map_nz - 1) ) {
					ddata[l] += Util::linear_interpolate(sdata[k], sdata[k + 1],t);
				}
				else if ( xx == (map_nx - 1) ) {
					ddata[l] += Util::bilinear_interpolate(sdata[k], sdata[k + map_nx], sdata[k + map_nxy], sdata[k + map_nxy + map_nx],u,v);
				}
				else if ( yy == (map_ny - 1) ) {
					ddata[l] += Util::bilinear_interpolate(sdata[k], sdata[k + 1], sdata[k + map_nxy], sdata[k + map_nxy + 1],t,v);
				}
				else if ( zz == (map_nz - 1) ) {
					ddata[l] += Util::bilinear_interpolate(sdata[k], sdata[k + 1], sdata[k + map_nx], sdata[k + map_nx + 1],t,u);
				}

//				if (k >= map->get_size()) {
//					cout << xx << " " << yy << " " <<  zz << " " << endl;
//					cout << k << " " << get_size() << endl;
//					cout << get_xsize() << " " << get_ysize() << " " << get_zsize() << endl;
//					throw;
//					}
//
//				ddata[l] = Util::trilinear_interpolate(sdata[k],
//						sdata[k + 1], sdata[k + map_nx],sdata[k + map_nx + 1],
//						sdata[k + map_nxy], sdata[k + map_nxy + 1], sdata[k + map_nx + map_nxy],
//						sdata[k + map_nx + map_nxy + 1],t, u, v);
			}
			else {
				if ( xx > map_nx - 1 || yy > map_ny - 1 || zz > map_nz - 1) {
					ddata[l] = 0;
					continue;
				}
				size_t k = Util::round(xx) + Util::round(yy) * map_nx + Util::round(zz) * (size_t)map_nxy;
				ddata[l] = sdata[k];
			}

		}
	}

	update();

	EXITFUNC;
}

EMData *EMData::unwrap_largerR(int r1,int r2,int xs, float rmax_f) {
	float *d,*dd;
	int do360=2;
	int rmax = (int)(rmax_f+0.5f);
	unsigned long i;
	unsigned int nvox=get_xsize()*get_ysize();//ming
	float maxmap=0.0f, minmap=0.0f;
	float temp=0.0f, diff_den=0.0f, norm=0.0f;
	float cut_off_va =0.0f;

	d=get_data();
	maxmap=-1000000.0f;
	minmap=1000000.0f;
	for (i=0;i<nvox;i++){
		if(d[i]>maxmap) maxmap=d[i];
		if(d[i]<minmap) minmap=d[i];
	}
	diff_den = maxmap-minmap;
	for(i=0;i<nvox;i++) {
		temp = (d[i]-minmap)/diff_den;
		if(cut_off_va != 0.0) {               // cut off the lowerset ?% noisy information
	 		if(temp < cut_off_va)
	   			d[i] = 0.0f;                   // set the empty part density=0.0
	 		else
	   			d[i] = temp-cut_off_va;
		}
		else	d[i] = temp;
	}

	for(i=0;i<nvox;i++) {
		temp=d[i];
		norm += temp*temp;
	}
	for(i=0;i<nvox;i++)		d[i] /= norm;                      //  y' = y/norm(y)

	if (xs<1) {
		xs = (int) floor(do360*M_PI*get_ysize()/4); // ming
		xs=Util::calc_best_fft_size(xs); // ming
	}
	if (r1<0) r1=0;
	float maxext=ceil(0.6f*std::sqrt((float)(get_xsize()*get_xsize()+get_ysize()*get_ysize())));// ming add std::

	if (r2<r1) r2=(int)maxext;
	EMData *ret = new EMData;

	ret->set_size(xs,r2+1,1);

	dd=ret->get_data();

	for (int i=0; i<xs; i++) {
		float si=sin(i*M_PI*2/xs);
		float co=cos(i*M_PI*2/xs);
		for (int r=0; r<=maxext; r++) {
			float x=(r+r1)*co+get_xsize()/2; // ming
			float y=(r+r1)*si+get_ysize()/2; // ming
			if(x<0.0 || x>=get_xsize()-1.0 || y<0.0 || y>=get_ysize()-1.0 || r>rmax){    //Ming , ~~~~ rmax need pass here
				for(;r<=r2;r++)                                   // here r2=MAXR
					dd[i+r*xs]=0.0;
        		break;
		    }
			int x1=(int)floor(x);
			int y1=(int)floor(y);
			float t=x-x1;
			float u=y-y1;
			float f11= d[x1+y1*get_xsize()]; // ming
			float f21= d[(x1+1)+y1*get_xsize()]; // ming
			float f12= d[x1+(y1+1)*get_xsize()]; // ming
			float f22= d[(x1+1)+(y1+1)*get_xsize()]; // ming
			dd[i+r*xs] = (1-t)*(1-u)*f11+t*(1-u)*f21+t*u*f22+(1-t)*u*f12;
		}
	}
	update();
	ret->update();
	return ret;
}


EMData *EMData::oneDfftPolar(int size, float rmax, float MAXR){		// sent MAXR value here later!!
	float *pcs=get_data();
	EMData *imagepcsfft = new EMData;
	imagepcsfft->set_size((size+2), (int)MAXR+1, 1);
	float *d=imagepcsfft->get_data();

	EMData *data_in=new EMData;
	data_in->set_size(size,1,1);
	float *in=data_in->get_data();

	for(int row=0; row<=(int)MAXR; ++row){
		if(row<=(int)rmax) {
			for(int i=0; i<size;++i)	in[i] = pcs[i+row*size]; // ming
			data_in->set_complex(false);
			data_in->do_fft_inplace();
			for(int j=0;j<size+2;j++)  d[j+row*(size+2)]=in[j];
		}
		else for(int j=0;j<size+2;j++) d[j+row*(size+2)]=0.0;
	}
	imagepcsfft->update();
	delete data_in;
	return imagepcsfft;
}

void EMData::uncut_slice(EMData * const map, const Transform& transform) const
{
	ENTERFUNC;

	if (!map) throw NullPointerException("NULL image");
	// These restrictions should be ultimately restricted so that all that matters is get_ndim() = (map->get_ndim() -1)
	if ( get_ndim() != 2 ) throw ImageDimensionException("Can not call cut slice on an image that is not 2D");
	if ( map->get_ndim() != 3 ) throw ImageDimensionException("Can not cut slice from an image that is not 3D");
	// Now check for complex images - this is really just being thorough
	if ( is_complex() ) throw ImageFormatException("Can not call cut slice on an image that is complex");
	if ( map->is_complex() ) throw ImageFormatException("Can not cut slice from a complex image");

// 	Transform3D r( 0, 0, 0); // EMAN by default
// 	if (!ort) {
// 		ort = &r;
// 	}

	float *ddata = map->get_data();
	float *sdata = get_data();

	int map_nx = map->get_xsize();
	int map_ny = map->get_ysize();
	int map_nz = map->get_zsize();
	int map_nxy = map_nx * map_ny;
	float map_nz_round_limit = (float) map_nz-0.5f;
	float map_ny_round_limit = (float) map_ny-0.5f;
	float map_nx_round_limit = (float) map_nx-0.5f;
/*
	Vec3f posttrans = ort->get_posttrans();
	Vec3f pretrans = ort->get_pretrans();*/

	int ymax = ny/2;
	if ( ny % 2 == 1 ) ymax += 1;
	int xmax = nx/2;
	if ( nx % 2 == 1 ) xmax += 1;
	for (int y = -ny/2; y < ymax; y++) {
		for (int x = -nx/2; x < xmax; x++) {
			Vec3f coord(x,y,0);
			Vec3f soln = transform*coord;
// 			float xx = (x+pretrans[0]) * (*ort)[0][0] +  (y+pretrans[1]) * (*ort)[0][1] + pretrans[2] * (*ort)[0][2] + posttrans[0];
// 			float yy = (x+pretrans[0]) * (*ort)[1][0] +  (y+pretrans[1]) * (*ort)[1][1] + pretrans[2] * (*ort)[1][2] + posttrans[1];
// 			float zz = (x+pretrans[0]) * (*ort)[2][0] +  (y+pretrans[1]) * (*ort)[2][1] + pretrans[2] * (*ort)[2][2] + posttrans[2];
//
// 			xx += map_nx/2;
// 			yy += map_ny/2;
// 			zz += map_nz/2;
//
			float xx = soln[0]+map_nx/2;
			float yy = soln[1]+map_ny/2;
			float zz = soln[2]+map_nz/2;

			// These 0.5 offsets are here because the round function rounds to the nearest whole number.
			if (xx < -0.5 || yy < -0.5 || zz < -0.5 || xx >= map_nx_round_limit || yy >= map_ny_round_limit || zz >= map_nz_round_limit) continue;

			size_t k = Util::round(xx) + Util::round(yy) * map_nx + Util::round(zz) * (size_t)map_nxy;
			int l = (x+nx/2) + (y+ny/2) * nx;
			ddata[k] = sdata[l];
		}
	}

	map->update();
	EXITFUNC;
}

EMData *EMData::extract_box(const Transform& cs, const Region& r)
{
	vector<float> cs_matrix = cs.get_matrix();
	
	EMData* box = new EMData();
	box->set_size((r.get_width()-r.x_origin()), (r.get_height()-r.y_origin()), (r.get_depth()-r.z_origin()));
	int box_nx = box->get_xsize();
	int box_ny = box->get_ysize();
	int box_nxy = box_nx*box_ny;
	float* bdata = box->get_data();
	float* ddata = get_data();
	
	for (int x = r.x_origin(); x < r.get_width(); x++) {
		for (int y = r.y_origin(); y < r.get_height(); y++) {
			for (int z = r.z_origin(); z < r.get_depth(); z++) {
				//float xb = cs_matrix[0]*x + cs_matrix[1]*y + cs_matrix[2]*z + cs_matrix[3];
				//float yb = cs_matrix[4]*x + cs_matrix[5]*y + cs_matrix[6]*z + cs_matrix[7];
				//float zb = cs_matrix[8]*x + cs_matrix[9]*y + cs_matrix[10]*z + cs_matrix[11];
				float xb = cs_matrix[0]*x + y*cs_matrix[4] + z*cs_matrix[8] + cs_matrix[3];
				float yb = cs_matrix[1]*x + y*cs_matrix[5] + z*cs_matrix[9] + cs_matrix[7];
				float zb = cs_matrix[2]*x + y*cs_matrix[6] + z*cs_matrix[10] + cs_matrix[11];
				float t = xb - Util::fast_floor(xb);
				float u = yb - Util::fast_floor(yb);
				float v = zb - Util::fast_floor(zb);
				
				//cout << x << " " << y << " " << z << " Box " << xb << " " << yb << " " << zb << endl;
				int l = (x - r.x_origin()) + (y - r.y_origin())*box_nx + (z - r.z_origin())*box_nxy;
				int k = (int) (Util::fast_floor(xb) + Util::fast_floor(yb) * nx + Util::fast_floor(zb) * nxy);
				//cout << k << " " << l << endl;
				if ( xb > nx - 1 || yb > ny - 1 || zb > nz - 1) {
					bdata[l] = 0;
					continue;
				}
				if (xb < 0 || yb < 0 || zb < 0){
					bdata[l] = 0;
					continue;
				}

				if (xb < (nx - 1) && yb < (ny - 1) && zb < (nz - 1)) {
					bdata[l] = Util::trilinear_interpolate(ddata[k], ddata[k + 1], ddata[k + nx],ddata[k + nx + 1], ddata[k + nxy], ddata[k + nxy + 1], ddata[k + nx + nxy], ddata[k + nx + nxy + 1],t, u, v);
				}
			}
		}
	}
	
	return box;
}

void EMData::save_byteorder_to_dict(ImageIO * imageio)
{
	string image_endian = "ImageEndian";
	string host_endian = "HostEndian";

	if (imageio->is_image_big_endian()) {
		attr_dict[image_endian] = "big";
	}
	else {
		attr_dict[image_endian] = "little";
	}

	if (ByteOrder::is_host_big_endian()) {
		attr_dict[host_endian] = "big";
	}
	else {
		attr_dict[host_endian] = "little";
	}
}

EMData* EMData::compute_missingwedge(float wedgeangle, float start, float stop)
{		
	EMData* test = new EMData();
	test->set_size(nx,ny,nz);
	
	float ratio = tan((90.0f-wedgeangle)*M_PI/180.0f);
	
	int offset_i = 2*int(start*nz/2);
	int offset_f = int(stop*nz/2);
	
	int step = 0;
	float sum = 0.0;
	double square_sum = 0.0;
	for (int j = 0; j < offset_f; j++){
		for (int k = offset_i; k < offset_f; k++) {
			for (int i = 0; i < nx; i+=2) {
				if (i < int(k*ratio)) {
					test->set_value_at(i, j, k, 1.0);
					float v = std::sqrt(pow(get_value_at_wrap(i, j, k),2) + pow(get_value_at_wrap(i+1, j, k),2));
					sum += v;
					square_sum += v * (double)(v);
					step++;
				}
			}
		}
	}
	
	float mean = sum / step;
	
	#ifdef _WIN32
	float sigma = (float)std::sqrt( _cpp_max(0.0,(square_sum - sum*mean)/(step-1)));
	#else
	float sigma = (float)std::sqrt(std::max<double>(0.0,(square_sum - sum*mean)/(step-1)));
	#endif	//_WIN32
	
	cout << "Mean sqr wedge amp " << mean << " Sigma Squ wedge Amp " << sigma << endl;
	set_attr("spt_wedge_mean", mean);
	set_attr("spt_wedge_sigma", sigma);
	
	return test;
}




	
	
