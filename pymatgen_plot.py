from pymatgen.analysis.adsorption import *


def plot_struct(slab, ax, scale=0.8, repeat=1, window=1.5, draw_unit_cell=True, decay=0.1, adsorption_sites=True):
    orig_slab = slab.copy()
    slab = reorient_z(slab)
    orig_cell = slab.lattice.matrix.copy()

    if repeat:
        slab.make_supercell([repeat, repeat, 1])
    coords = np.array(sorted(slab.cart_coords, key=lambda x: x[2]))  # Cart coordinates
    sites = sorted(slab.sites, key=lambda x: x.coords[2])
    corner = [0, 0, slab.lattice.get_fractional_coords(coords[-1])[-1]]
    corner = slab.lattice.get_cartesian_coords(corner)[:2]
    alphas = 1 - decay * (np.max(coords[:, 2]) - coords[:, 2])
    alphas = alphas.clip(min=0)

    def plot_view(view):
        """
        Args:
            view(str): top, sidef, sidel
        """
        nonlocal alphas
        if view == "top":
            vector = [0, 1]
        elif view == "sidef":
            vector = [1, 2]
        elif view == "sidel":
            vector = [0, 2]
        else:
            vector = None

        index = [[ii[0] for ii in itertools.product(vector, vector)],
                 [ii[1] for ii in itertools.product(vector, vector)]]
        verts = orig_cell[tuple(index)].reshape(2, 2)
        lattsum = verts[0] + verts[1]
        lattsum[1] = 0 if view != "top" else lattsum[1]

        # Draw circles at sites and stack them accordingly
        for n, coord in enumerate(coords):
            r = sites[n].specie.atomic_radius * scale
            ax.add_patch(patches.Circle(coord[vector] - lattsum * (repeat // 2),
                                        r, color='w', zorder=2 * n))
            color = color_dict[sites[n].species_string]
            alphas[n] = 1.0 if view != "top" else alphas[n]
            ax.add_patch(patches.Circle(coord[vector] - lattsum * (repeat // 2), r,
                                        facecolor=color, alpha=alphas[n],
                                        edgecolor='k', lw=0.3, zorder=2 * n + 1))
        # Adsorption sites
        if adsorption_sites and view == "top":
            asf = AdsorbateSiteFinder(orig_slab)
            ads_sites = asf.find_adsorption_sites()['all']
            sop = get_rot(orig_slab)
            ads_sites = [sop.operate(ads_site)[:2].tolist()
                         for ads_site in ads_sites]
            ax.plot(*zip(*ads_sites), color='k', marker='x',
                    markersize=10, mew=1, linestyle='', zorder=10000)

        # Draw unit cell
        if draw_unit_cell:
            lattsum = verts[0] + verts[1]
            verts = np.insert(verts, 1, lattsum, axis=0).tolist()
            verts += [[0., 0.]]
            verts = [[0., 0.]] + verts
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO,
                     Path.LINETO, Path.CLOSEPOLY]
            verts = [(np.array(vert) + corner).tolist() for vert in verts]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2,
                                      alpha=0.5, zorder=2 * n + 2)
            ax.add_patch(patch)

        ax.set_aspect("equal")
        center = corner + lattsum / 2.
        extent = np.max(lattsum)
        lim_array = [center - extent * window, center + extent * window]
        x_lim = [ele[0] for ele in lim_array] if view == "top" else [center[0] - 2 * center[0] * window,
                                                                     center[0] + 2 * center[0] * window]
        y_lim = [ele[1] for ele in lim_array] if view == "top" else [-1, extent + 1]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        return ax

    return plot_view
