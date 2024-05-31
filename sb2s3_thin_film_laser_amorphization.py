# -*- coding: utf-8 -*-

# Josef Resl 2024. Licensed under the MIT License.
# See LICENSE for details.



"""
Picosecond laser melting simulatinon of a thin film

This script simulates the heating and phase change dynamics in thin films induced by pulsed laser irradiation. 
The simulation integrates  thin film interference, heat conduction, and temperature-dependent optical properties  and melting induced changes in the refractive index.

Overview:
- This simulation is designed to study and optimize the amorphization process of Sb2S3 thin films under laser pulses.
- It accounts for the interference and absorption of laser light within the thin film and the resulting localized heating and heat conduction.
- The model includes the temperature-dependent changes in material properties, such as absorption coefficient and specific heat capacity, and incorporates the melting and solidification phases of the material.

Key Physical Phenomena:
1. Thin Film Interference: The interaction of laser light with the thin film stack, leading to interference patterns that affect the distribution of the electric field and absorption within the film.
2. Heat Conduction: The spread of heat through the material, modeled primarily in the direction normal to the film surface, assuming uniform heating in the plane.
3. Temperature-Dependent Properties: Changes in optical and thermal properties of the materials with temperature, influencing the dynamics of heating and cooling.
4. Melting and Solidification: The transition of the material from solid to liquid upon heating and back to solid upon cooling, including the specific heat of melting and the formation of mixed-phase regions.

Simulation Workflow:
- The locally absorbed power is calculated using the transfer matrix method.
- The temperature distribution is updated based on the absorbed power and heat conduction.
- The optical and thermal properties are recalculated for the next time step based on the updated temperature.
- This iterative process continues to simulate the entire heating and cooling cycle induced by the laser pulse.

Example Usage:
    >>> run_simulation(end=1.389e-9, a1=1, a2=1)
This command runs the simulation for a specified energy density, fitting parameters a1 and a2, and generates plots of the results if enabled.

"""


import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, log, mesh, nls
from mpi4py import MPI
from petsc4py import PETSc
from scipy import constants
from scipy.special import erf, erfc


def run_simulation(end, a1=1, a2=1, plot=True, printt=False):
    # a1,a2 are for fitting and optimisation when needed

    # custom print function, that can be disabled
    def pprint(*args, **kwargs):
        if printt:

            print(*args, **kwargs)

    pi = np.pi

    zert = 273.15  # absolute zero
    rt = zert + 22  # room temp in kelvins

    # Define temporal parameters
    t = 0  # Start time
    tmaxx = 33e-12  # Final time

    dt = 0.1e-12  # timestep
    dtc = 1000 * dt  # cooling tmestep

    tp = 33e-12  # laser pulse duration

    num_steps = int(tmaxx / dt)

    nm = 1e-9

    tempdep = "YES"  # temperature dependent interference
    # tempdep = "NO"

    smelting = "YES"  # melting
    # smelting="NO"

    cooling = "YES"  # cooling
    cooling = "NO"

    plotonly = True  #  plotting

    plotonly = False  # simulation and plotting

    # material proerties

    # silicon epsilon temperature dependent
    epr = [1.75085352e-06, 3.50405339e-03, 1.71065514e01]
    epim = [4.71799785e-06, -9.51075383e-04, 1.73939034e-01]

    def ntsi(T=rt):  # equtionn only valid up pto 500 deg C
        T = T - zert  # celsius
        eps = (
            epr[2]
            + epr[1] * T
            + epr[0] * T**2
            + (epim[2] + epim[1] * T + epim[0] * T**2) * 1j
        )
        return eps**0.5

    def ntsb2s3(T=rt):  # refractive index of crystalline sb2s3

        return (3.77 + 0.654j) * (T * 0 + 1)

    # heat capacity solid and liquid of sb2s3 with heat of melting with gaussian smoothing
    def cpsb2s3(T=rt, s=1):
        T = np.where(T > 1420, 1420, T)

        # in kelvins
        hc = (0.3989422804014327 * (120223.0 - 0.307035 * s**2)) * (
            2.718281828459045 ** (-(0.5 * (-Sb2S3["tmk"] + T) ** 2) / s**2)
        ) / s + 0.5 * (
            (844.035 - 0.307035 * T)
            * (1.0 + erf((0.7071067811865475 * (-Sb2S3["tmk"] + T)) / s))
            + 336.0 * erfc((0.7071067811865475 * (-Sb2S3["tmk"] + T)) / s)
        )
        return 1 * hc

    def cpsi(T=22, s=1):  # silicon heat capacity
        return 703

    # dicts for material properties in SI units

    Si = {  # substrate
        "r": 2330,  # mass density
        "k": 149,  # thermal conductivity
        "c": cpsi,  # heat capacity
        "n": ntsi,  # refractive index - temperature dependent
        "tmk": 1687,  # melting temperature in K
        "mh": 1803809,  # heat of melting
        "pcm": 0,  # is a pcm or not
    }  # wafer single cryst.

    Sb2S3 = {  # film
        "r": 4380,
        "k": 0.216,
        "c": cpsb2s3,
        "n": ntsb2s3,  # refractive index crystalline  phase
        "na": 2.33 + 0.0479j,  # refractive index amorphous phase
        "tmk": 823,
        "mh": 120223,
        "pcm": 1,
    }

    # check thermal diffusion length

    # Call the cpsb2s3 function with T_example and s_example to get the heat capacity
    heat_capacity = Sb2S3["c"](T=rt, s=1)

    # Corrected pprint statement with a comma and correct function call for heat capacity
    pprint(
        "thermal diffusion length",
        (4 * Sb2S3["k"] / (heat_capacity * Sb2S3["r"]) * 0.5 * tp) ** 0.5,
    )

    # bruggeman effective medium approximation
    def ema(
        na, nb, fa, L=1 / 3.0
    ):  # anisotropic bruggemann EMA depolarisation factor = L, 0, disks, 1 needles
        L = 0.9999
        epa = na**2
        epb = nb**2
        epema = (
            epa * (-fa + L)
            + epb * (-1 + fa + L)
            - np.sqrt(
                -4 * epa * epb * (-1 + L) * L
                + (epa * (-fa + L) + epb * (-1 + fa + L)) ** 2
            )
        ) / (2.0 * (-1 + L))
        return epema**0.5

    def sm(ar):  # smoothing material properties at film/substrate
        # interface if necessary by convolution
        ds = 0 * 5  # smoothing width in nm
        sw0 = round((ds / dx + 3) / 4.0)
        pprint(sw0)

        ker0 = np.arange(sw0) * 0 + 1 / sw0

        ker = np.convolve(ker0, ker0, mode="full")  # triangle
        ker = np.convolve(ker, ker, mode="full")  # cubic kernel
        sw = ker.size
        pprint("smoothing", sw * dx)

        arp = np.pad(ar, sw, mode="edge")
        av = np.convolve(arp, ker, mode="same")
        # return ar
        return av[sw + 0 : av.size - sw + 0]

    # Stack of materials and their thicknesses in nm
    stack = [Sb2S3, Si]
    fth = 109 * 2  # film thickness
    thicknesses = [fth, 400]  # simulate 400 nm of subtrate too

    # Define the grid spacing
    grid_spacing = 0.5  # more than 1000 points are slow
    # material properties and layers

    # Get the unique materials in the stack
    # unique_materials = np.unique(stack) # not needed for single occurence of material

    # Calculate the total thickness of the stack
    total_thickness = sum(thicknesses)

    # Generate the grid points
    nx = int(total_thickness / grid_spacing)
    x = np.linspace(0, total_thickness, nx + 1)  # needed?

    # FEM DOMAIN
    domain = mesh.create_interval(
        MPI.COMM_WORLD, nx, np.array([0, total_thickness * nm])
    )
    dx = x[1] - x[0]

    # Initialize an empty list to store the arrays for each material
    material_arrays = []

    # Material properties
    k = 0 * x
    r = 0 * x
    c = 0 * x
    pcm = 0 * x

    # Loop through each unique material
    for material in stack:
        # Initialize a binary array for the material's presence in the stack
        material_array = np.zeros(len(x))

        # Loop through the stack and update the material array
        current_thickness = 0
        for m, thickness in zip(stack, thicknesses):
            if m == material:
                start_index = int(current_thickness / grid_spacing)
                end_index = int((current_thickness + thickness) / grid_spacing)
                material_array[start_index:end_index] = 1
                # smooth
                material_array = sm(material_array)
                # create arrays of thermal properties
                k = k + material_array * material["k"]
                r = r + material_array * material["r"]
                c = c + material_array * material["c"]()
                pcm = pcm + material_array * material["pcm"]

            current_thickness += thickness

        # Append the material array to the list
        material_arrays.append(material_array)

    lm = stack[-1]  # fix last entry

    k[-1] = lm["k"]
    r[-1] = lm["r"]
    c[-1] = lm["c"]()
    pcm[-1] = lm["pcm"]
    material_arrays[-1][-1] = 1

    c0 = c  # for later -> melting

    ## interference

    from tmm_core_numba import (  # numba acceleration for slowest functions; from tmm_core import (
        coh_tmm, find_in_structure_with_inf, position_resolved)

    def interf():  # position resolved, at room temperature

        # n,d_list must be numpy arrays

        # thicknesses

        d_list = np.insert(np.array([np.inf, np.inf]), 1, thicknesses[:-1])  # in nm

        # room temp
        # opt properties

        n_list = np.insert(np.array([1, Si["n"]()], dtype=complex), 1, Sb2S3["n"]())

        th_0 = 0  # angle of incidence
        lam_vac = 532  # wavelength
        pol = "p"  # poölarisation

        # interference

        (
            r,
            t,
            R,
            T,
            power_entering,
            vw_list,
            kz_list,
            th_list,
            pol,
            n_list,
            d_list,
            th_0,
            lam_vac,
        ) = coh_tmm(pol, n_list, d_list, th_0, lam_vac)

        ds = x  # position in structure
        poyn = []
        absor = []

        # calculate inside layers

        for d in ds:
            layer, d_in_layer = find_in_structure_with_inf(d_list, d)
            poynt, absort, _, _, _ = position_resolved(
                int(layer),
                d_in_layer,
                r,
                t,
                R,
                T,
                power_entering,
                vw_list,
                kz_list,
                th_list,
                pol,
                n_list,
                d_list,
                th_0,
                lam_vac,
            )
            poyn.append(poynt)
            absor.append(absort)
        # convert data to numpy arrays for convenience
        poyn = np.array(poyn)
        absor = np.array(absor)

        return (absor, poyn)

    def interftemp(nt):  # position resolved, temperature dependent calculation

        # same as above except for list of current refractive indices nt

        # every gridpoint is an optical layer..
        d_list = np.insert(np.array([np.inf, np.inf]), 1, x * 0 + dx)  # in nm

        n_list = np.insert(np.array([1, Si["n"]()], dtype=complex), 1, nt)

        th_0 = 0  # pi/4
        lam_vac = 532
        pol = "p"
        (
            r,
            t,
            R,
            T,
            power_entering,
            vw_list,
            kz_list,
            th_list,
            pol,
            n_list,
            d_list,
            th_0,
            lam_vac,
        ) = coh_tmm(pol, n_list, d_list, th_0, lam_vac)

        # print(R)

        ds = x
        poyn = []
        absor = []
        for d in ds:
            layer, d_in_layer = find_in_structure_with_inf(d_list, d)
            poynt, absort, _, _, _ = position_resolved(
                int(layer),
                d_in_layer,
                r,
                t,
                R,
                T,
                power_entering,
                vw_list,
                kz_list,
                th_list,
                pol,
                n_list,
                d_list,
                th_0,
                lam_vac,
            )
            poyn.append(poynt)
            absor.append(absort)
        # convert data to numpy arrays for convenience
        # poyn = np.array(poyn)
        absor = np.array(absor)

        return absor

    # energy density of the gaussian profile at radius r and height z
    # center is r=0
    # this is for 50% amplification of the pumping laser

    def edg(r, z):  # r in um , z in mm # new fit
        a = 0.00020452525765315686  # Joules
        s0 = 136.17234300854295  # um
        div = -12.569358565114273  # um
        return a / (
            2.0
            * np.exp(1) ** (r**2 / (2.0 * ((s0 + div * (z - 12.5)) ** 2)))
            * np.pi
            * ((s0 + div * (z - 12.5)) ** 2)
        )

    ed = end

    pprint(ed, "Energy per um^2 at chosen r and z")

    pwr = ed * 10 ** (6 + 6) / (33 * 10 ** (-12)) * 10**9  # power per cubic meter

    absor, poyn = interf()  # absorption at room temp

    p = pwr * absor  # initial absorbed power

    absor0 = absor

    ntold = 0 * x  # previous refractive index

    stimes = -1 * np.ones(num_steps)  # store solver times
    mofplo = np.outer(np.zeros(num_steps), 0 * x)  # store molten fraction

    # initialize optical properties
    for matarr, mat in zip(material_arrays, stack):
        ntold = ntold + matarr * mat["n"]()

    # simulation time loop and preparation therof

    # Create initial condition of temperature at room temp
    def initial_condition(x):
        return 0 * x[0] + rt

    V = fem.FunctionSpace(domain, ("CG", 1))

    ## material properties

    # absorbed power heating the  layers
    f = fem.Function(V)
    f.vector[:] = p

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)

    # Define solution variable, and interpolate initial condition
    uh = fem.Function(V)
    uh.name = "uh"
    uh.interpolate(initial_condition)

    # auxilary variable for bondary condition deep in substrate
    ub = fem.Function(V)
    ub.vector[:] = c * 0 + rt

    # theremal material properties
    kappa = fem.Function(V)  # heat cond
    kappa.vector[:] = k

    cp = fem.Function(V)  # heat cap
    cp.vector[:] = c

    ro = fem.Function(V)  # mass density
    ro.vector[:] = r

    # boundary conditions
    bchack = fem.Function(V)  # a bit of a hack to set one sided boundary condition
    bchack.vector[:] = np.where(x > np.mean(x), 1, 0)

    sol = np.zeros([num_steps, nx + 1])  # store solutions here

    v = ufl.TestFunction(V)

    # some loop vars
    ii = 1  # no of inter ference calcs
    iii = 0  # timesteps
    i = 0  # timesteps

    # time stepping loop
    while t < tmaxx:
        s = 20  # smoothing parameter for melting
        if t <= tp:  # advance time
            t += dt
        elif t > tp and cooling == "YES":  # larger timestep for cooling
            t += dtc
            dt = dtc
        else:
            break

        stimes[i] = t

        pprint(f"\r" + str(round(100 * t / tp)) + " %", end="")

        if smelting == "YES":  # adjust the heat capacity when melting
            cc = 0 * c0

            for matarr, mat in zip(material_arrays, stack):
                cc = cc + matarr * mat["c"](uh.x.array, s)

            cp.vector[:] = cc

        if tempdep == "YES":
            # calulate molten fraction in every point
            pcm = cc * 0
            for matarr, mat in zip(material_arrays, stack):
                pcm = pcm + matarr * mat["pcm"]

            fm = (1 + erf((uh.x.array - Sb2S3["tmk"]) / (np.sqrt(2) * s))) / 2
            fm = fm * pcm

            mofplo[iii, :] = fm

            # interference

            nt = 0 * x

            # prepare rfractive index array

            for matarr, mat in zip(material_arrays, stack):
                nn = matarr * mat["n"](uh.x.array)
                if mat["pcm"] == 1:  # if PCM
                    nn = matarr * ema(mat["n"](uh.x.array), mat["na"], 1 - fm)
                nt = nt + nn

            if t > 33e-12:  # laser pulse off
                f.vector[:] = 0 * absor0

            # if max. change in refractive index is small,
            # dont recalulate interference to save time
            if np.amax(np.abs(ntold - nt) / ntold) > 0.005 and t <= tp:

                absor = interftemp(nt)
                f.vector[:] = pwr * absor
                # pprint("FFF")

                ii = ii + 1
                ntold = nt

            iii = iii + 1

        # Define Robin boundary condition "natural boundary condition"
        al = (
            0 * 4 * pi / (532) * stack[-1]["n"](sol[i - 1, -1]).imag
        )  # substrate absorption coeff

        h = fem.Constant(domain, al)  # absorbtion coeff
        T_ambient = fem.Constant(domain, rt)  # Ambient temperature

        # assemble heat eqution in weak form time with time discretisation
        F = (
            ro * cp * uh * v * ufl.dx
            + dt * ufl.dot(kappa * ufl.grad(uh), ufl.grad(v)) * ufl.dx
            - (ro * cp * u_n + dt * f) * v * ufl.dx
            # robin boundary condition,
            + bchack * h * (uh - T_ambient) * v * ufl.ds
        )

        # solver and parameters
        problem = fem.petsc.NonlinearProblem(F, uh)
        # problem = fem.petsc.NonlinearProblem(F, uh)

        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-12
        solver.report = True
        solver.max_it = 10**2

        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}pc_type"] = "gamg"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()

        # log.set_log_level(log.LogLevel.INFO)

        # Solve  problem
        solver.solve(uh)
        uh.x.scatter_forward()

        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array
        sol[i, :] = uh.x.array
        i += 1

    # last temp
    maxtemp = np.amax(sol[i - 1, :])

    def convert_time(time_in_seconds):
        # converts time in s to time with Si prefix
        # Define the units and their corresponding prefixes
        units = ["s", "ms", "us", "ns", "ps"]
        # The step size when moving between units
        step = 1_000

        # Start at the base unit (seconds)
        unit_index = 0
        while time_in_seconds < 1 and unit_index < len(units) - 1:
            time_in_seconds *= step
            unit_index += 1

        # Convert the time to a string with 3 significant digits
        time_str = "{:.3g}".format(time_in_seconds)
        return time_str + " " + units[unit_index]

    pprint(ii, "interference calcs")

    pprint(iii, "timesteps in hook")

    moltenper = 100 * np.trapz(mofplo[iii - 1, :], x) / fth

    pprint(moltenper, " % molten")
    pprint(maxtemp, " K max. Temp")

    if plot:

        ##### PLOTTING

        # selects some quasi-equdistant curves an plots them in a nice way
        import matplotlib.colors as mcolors

        # Define a colormap that goes from blue to red
        colors = ["blue", "red"]
        cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

        fig, axs = plt.subplots(2)

        total_indices = len(sol)
        dat = sol

        # automatic avarge temperature equidistant curves

        avtemp = np.zeros(len(dat))

        for i in range(len(dat)):
            avtemp[i] = np.mean(dat[i, : int(fth / grid_spacing)])

        # Determine the total temperature range
        min_temp = np.min(avtemp)
        max_temp = np.max(avtemp)

        interval_borders = np.linspace(min_temp, max_temp, 10)

        # Determine indices of interval borders
        # For simplicity, we find the closest average temperature to each interval border
        indices = np.searchsorted(avtemp, interval_borders)

        indices = np.clip(indices, 0, len(avtemp) - 1)
        indices = np.unique(indices)

        gradient = np.linspace(0, 1, indices.size)
        colors = cmap(gradient)

        dtt = dt
        ii = 0
        for i in indices:
            t = i * dtt

            axs[0].plot(x, dat[i, :], color=colors[ii], label=convert_time(t))
            ii = ii + 1

        # for i in range(0, int(tmaxx / dtt) + 0*int(tmaxx / dtt / ncurves), int(tmaxx / dtt / ncurves)):
        ii = 0
        for i in indices:
            t = i * dtt
            pprint(i, t)
            closest_index = (np.abs(stimes - t)).argmin()

            # axs[1].plot(x, powplo[:,closest_index], color=colors[int(i/(int(tmaxx / dtt / ncurves)))],label=f"{t:.0f} ps")
            axs[1].plot(
                x, mofplo[closest_index, :], color=colors[ii], label=convert_time(t)
            )
            ii = ii + 1

        # Function to create a colormap that fades from a specified color to white
        def fade_to_white(color, n=100):
            # Create an RGBA tuple for the base color
            base_color = mcolors.to_rgba(color)
            # Generate a list of colors that fade to white
            return [
                tuple(
                    base_color[i] + (1 - base_color[i]) * (j / (n - 1))
                    for i in range(3)
                )
                + (1,)
                for j in range(n)
            ]

        def clear_collections(ax):
            for coll in ax.collections:
                coll.remove()

        # Create the custom colormap
        n_gradient = 100
        gray_to_white = mcolors.LinearSegmentedColormap.from_list(
            "fade_to_white", fade_to_white("gray", n_gradient)
        )
        xplim = 380

        # Draw the gradient background using a loop and axvspan
        for ax in axs:
            # Clear existing collections using the custom function
            clear_collections(ax)

            # Add new gradient background
            for i in range(int(fth), int(xplim)):

                ax.axvspan(
                    i,
                    i + 1,
                    facecolor=gray_to_white((i - fth) / (xplim - fth)),
                    alpha=1,
                )

        # Hide the top and right spines
        for ax in axs:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        axs[0].spines["bottom"].set_visible(False)

        xx = x[int(fth / grid_spacing) + 20 :]
        # ax1.plot(xx,0*xx+813)
        # ax1.plot(xx,0*xx+823,label="Melting")

        axs[0].axvspan(0, fth, facecolor=(0.914, 0.729, 0.82))
        ##	axs[0].axvspan(fth, fth+300, facecolor='gray', alpha=0.2)
        # axs[1].axvspan(0, fth, facecolor='brown', alpha=0.2)
        # axs[1].axvspan(fth, fth+300, facecolor='gray', alpha=0.2)
        axs[1].axvspan(0, fth, facecolor=(0.914, 0.729, 0.82))
        ##	axs[1].axvspan(fth, fth+300, facecolor='gray', alpha=0.2)

        # axs[0].set_xlabel("depth [nm]")
        axs[0].set_ylabel("Temperature in K", color="red", fontweight="bold")
        axs[0].set_xlim(0, xplim)
        axs[0].set_ylim(250, 1500)
        axs[0].get_xaxis().set_visible(False)  # no axis

        axs[1].set_xlabel("Depth in nm")
        # axs[1].set_ylabel("Absorbed power fraction\nper nm", color=color,fontweight="bold")
        axs[1].set_ylabel("Local Molten Fraction", color="red", fontweight="bold")
        axs[1].set_xlim(0, xplim)
        # plt.title("Evolution of Temperature Distribution")

        # # Add label for film
        # axs[0].text(0.18, 0.99, '''Sb$_2$S$_3$
        # 109 nm''', transform=axs[0].transAxes, fontsize=12,  rotation=-45 ,verticalalignment='top' )

        # # Add label for substrate
        # axs[0].text(0.42, 0.85, "Si", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

        # Add label for film
        axs[0].text(
            0.18,
            0.99,
            """Sb$_2$S$_3$
218 nm""",
            transform=axs[0].transAxes,
            fontsize=12,
            rotation=-45,
            verticalalignment="top",
        )

        # Add label for substrate
        axs[0].text(
            0.8,
            0.3,
            "Si",
            transform=axs[0].transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        # Get handles and labels from the figure.
        handles, labels = axs[0].get_legend_handles_labels()

        # Reverse the order of the handles and labels.
        handles, labels = handles[::-1], labels[::-1]

        axs[0].legend(handles, labels, ncol=2)
        # 	fig.subplots_adjust(hspace=.05) # no whitespace between subplots
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig("my_figure.png", dpi=600, bbox_inches="tight", pad_inches=0)

        plt.show()

    return moltenper / 100.0


if __name__ == "__main__":
    run_simulation(end=1.389e-9, a1=1, a2=1)  # spot 16
