/**
 * $Id: imageio.cpp,v 1.30 2010/03/22 19:47:18 gtang Exp $
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

#include <cmath>
#include <cstdio>

#ifdef _WIN32
	#include <cfloat>
#endif	//_WIN32

#include "imageio.h"
#include "geometry.h"

using namespace EMAN;

ImageIO::~ImageIO()
{
}

int ImageIO::read_ctf(Ctf &, int)
{
	return 1;
}

void ImageIO::write_ctf(const Ctf &, int)
{

}

void ImageIO::check_region(const Region * area, const FloatSize & max_size,
						   bool is_new_file,bool inbounds_only)
{
	if (area) {
		if (is_new_file) {
			throw ImageReadException("", "file must exist before accessing its region");
		}
		int img_ndim = max_size.get_ndim();
		int area_ndim = area->get_ndim();

		if (area_ndim > img_ndim) {
			char desc[256];
			sprintf(desc, "Image is %dD. Cannot read %dD region", img_ndim, area_ndim);
			throw ImageReadException("", desc);
		}

		// EMUtil::process_region_io handles regions that are outside the image. So some image types don't mind if the
		// region is beyond the boundary. It would be ideal if they all could do this, but it would take some work.
		if (inbounds_only ){
			if (!area->is_region_in_box(max_size)) {
				char desc[1024];
				sprintf(desc, "Region box %s is outside image area (%d,%d,%d)",
						area->get_string().c_str(), (int)max_size[0],
						(int)max_size[1], (int)max_size[2]);
				throw ImageReadException("", desc);
			}
		}
	}
}

void ImageIO::check_region(const Region * area, const IntSize & max_size,
						   bool is_new_file, bool inbounds_only)
{
	check_region(area, FloatSize(max_size[0], max_size[1], max_size[2]),
				 is_new_file,inbounds_only);
}

void ImageIO::check_read_access(int image_index)
{
	init();

	int nimg = get_nimg();
	if (image_index < 0 || image_index >= nimg) {
		throw OutofRangeException(0, nimg-1, image_index, "image index");
	}
}

void ImageIO::check_read_access(int image_index, const float *data)
{
	check_read_access(image_index);
	if (!data) {
		throw NullPointerException("image data is NULL");
	}
}

void ImageIO::check_write_access(IOMode iomode, int image_index, int max_nimg)
{
	init();

	if (iomode == READ_ONLY) {
		throw ImageWriteException("", "File is not openned to write");
	}

	if ((image_index < -1) || (max_nimg > 0 && image_index >= max_nimg)) {
		throw OutofRangeException(-1, max_nimg - 1, image_index, "image index");
	}
}

void ImageIO::check_write_access(IOMode iomode, int image_index,
								 int max_nimg, const float *data)
{
	check_write_access(iomode, image_index, max_nimg);
	if (!data) {
		throw NullPointerException("image data is NULL");
	}
}

FILE *ImageIO::sfopen(const string & filename, IOMode mode,
					  bool * is_new, bool overwrite)
{
	FILE *f = 0;
	if (mode == READ_ONLY) {
		f = fopen(filename.c_str(), "rb");
	}
	else if (mode == READ_WRITE) {
		if (overwrite) {
			f = fopen(filename.c_str(), "wb");
			if (is_new) {
				*is_new = true;
			}
		}
		else {
			f = fopen(filename.c_str(), "r+b");
			if (!f) {
				FILE *f1 = fopen(filename.c_str(), "wb");
				if (!f1) {
					throw FileAccessException(filename);
				}
				else {
					if (is_new) {
						*is_new = true;
					}
					fclose(f1);
					f1 = 0;
					f = fopen(filename.c_str(), "r+b");
				}
			}
		}
	}
	else if (mode == WRITE_ONLY) {
		f = fopen(filename.c_str(), "wb");
		if (is_new) {
			*is_new = true;
		}
	}

	if (!f) {
		throw FileAccessException(filename);
	}
	return f;
}


int ImageIO::get_nimg()
{
	init();
	return 1;
}
