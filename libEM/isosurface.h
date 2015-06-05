/**
 * $Id: isosurface.h,v 1.13 2011/12/27 19:27:51 john Exp $
 */

/*
 * Author: Tao Ju, 5/16/2007 <taoju@cs.wustl.edu>, code ported by Grant Tang
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

#ifndef _ISOSURFACE_H_
#define _ISOSURFACE_H_

#include "emdata.h"

namespace EMAN
{

	class Isosurface {
	public:
		Isosurface() : _emdata(0), _surf_value(1) {}
		virtual ~Isosurface(){}

		/**
		 * Sets Voxel data for Isosurface implementation
		 */
		virtual void set_data(EMData* data) {
			_emdata = data;
		}

		/**
		 * Set Isosurface value
		 */
		virtual void set_surface_value(const float value) = 0;

		virtual float get_surface_value() const = 0;

		/**
		 * Set Grid Size
		 */
		virtual void set_sampling(const int size) = 0;

		virtual int get_sampling() const = 0;

		/** Get the number of feasible samplings
		*
		 */
		virtual int get_sampling_range() = 0;

		virtual Dict get_isosurface()  = 0;
		
		virtual void setRGBorigin(int x, int y, int z) = 0;
		
		virtual void setRGBscale(float i, float o) = 0;
		
		virtual void setRGBmode(int mode) = 0;
		
		virtual void setCmapData(EMData* data) = 0;
		
		virtual void setCmapMinMax(float min, float max) = 0;

	protected:
		EMData * _emdata;

		float _surf_value;
	};

}

#endif
