/**
 * $Id: fitsio.h,v 1.3 2008/03/24 01:22:22 gtang Exp $
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

#ifndef eman__fitsio_h__
#define eman__fitsio_h__ 1

#include "imageio.h"

namespace EMAN
{
	/** MRC file = header + data (nx x ny x nz).
	 * A MRC image file stores 1D, 2D or 3D image. The image's
	 * dimensions and pixel type are defined in the header.
	 */
	
	class FitsIO:public ImageIO
	{
	public:
		explicit FitsIO(const string & filename, IOMode rw_mode = READ_ONLY);
		~FitsIO();

		DEFINE_IMAGEIO_FUNC;

		int read_ctf(Ctf & ctf, int image_index = 0);
		void write_ctf(const Ctf & ctf, int image_index = 0);

		static bool is_valid(const void *first_block, off_t file_size = 0);
		static int get_mode_size(int mm);
		static int to_em_datatype(int mrcmode);
		static int to_mrcmode(int em_datatype, int is_complex);

	private:
		string filename;
		IOMode rw_mode;
		FILE *fitsfile;

		bool is_big_endian;
		bool is_new_file;
		bool initialized;
		int dstart;
		int dtype;
		int nx,ny,nz;
	};
}

#endif	//eman__mrcio_h__
