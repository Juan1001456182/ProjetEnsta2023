#include <SFML/Window/Keyboard.hpp>
#include <ios>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include "cartesian_grid_of_speed.hpp"
#include "vortex.hpp"
#include "cloud_of_points.hpp"
#include "runge_kutta.hpp"
#include "screen.hpp"

auto readConfigFile( std::ifstream& input )
{
    using point=Simulation::Vortices::point;

    int isMobile;
    std::size_t nbVortices;
    Numeric::CartesianGridOfSpeed cartesianGrid;
    Geometry::CloudOfPoints cloudOfPoints;
    constexpr std::size_t maxBuffer = 8192;
    char buffer[maxBuffer];
    std::string sbuffer;
    std::stringstream ibuffer;
    // Lit la première ligne de commentaire :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer);// Lecture de la grille cartésienne
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    double xleft, ybot, h;
    std::size_t nx, ny;
    ibuffer >> xleft >> ybot >> nx >> ny >> h;
    cartesianGrid = Numeric::CartesianGridOfSpeed({nx,ny}, point{xleft,ybot}, h);
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit mode de génération des particules
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    int modeGeneration;
    ibuffer >> modeGeneration;
    if (modeGeneration == 0) // Génération sur toute la grille 
    {
        std::size_t nbPoints;
        ibuffer >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {cartesianGrid.getLeftBottomVertex(), cartesianGrid.getRightTopVertex()});
    }
    else 
    {
        std::size_t nbPoints;
        double xl, xr, yb, yt;
        ibuffer >> xl >> yb >> xr >> yt >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {point{xl,yb}, point{xr,yt}});
    }
    // Lit le nombre de vortex :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le nombre de vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    try {
        ibuffer >> nbVortices;        
    } catch(std::ios_base::failure& err)
    {
        std::cout << "Error " << err.what() << " found" << std::endl;
        std::cout << "Read line : " << sbuffer << std::endl;
        throw err;
    }
    Simulation::Vortices vortices(nbVortices, {cartesianGrid.getLeftBottomVertex(),
                                               cartesianGrid.getRightTopVertex()});
    input.getline(buffer, maxBuffer);// Relit un commentaire
    for (std::size_t iVortex=0; iVortex<nbVortices; ++iVortex)
    {
        input.getline(buffer, maxBuffer);
        double x,y,force;
        std::string sbuffer(buffer, maxBuffer);
        std::stringstream ibuffer(sbuffer);
        ibuffer >> x >> y >> force;
        vortices.setVortex(iVortex, point{x,y}, force);
    }
    input.getline(buffer, maxBuffer);// Relit un commentaire
    input.getline(buffer, maxBuffer);// Lit le mode de déplacement des vortex
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    ibuffer >> isMobile;
    return std::make_tuple(vortices, isMobile, cartesianGrid, cloudOfPoints);
}

int main( int nargs, char* argv[] )
{   

    //Begin MPI process
    int num_procs;
    int rank;

    MPI_Init(&nargs, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("%d: hello (p=%d)\n", rank, num_procs);
    MPI_Status status;

    
    char const* filename;
    if (nargs==1)
    {
        std::cout << "Usage : vortexsimulator <nom fichier configuration>" << std::endl;
        return EXIT_FAILURE;
    }

    filename = argv[1];
    std::ifstream fich(filename);
    auto config = readConfigFile(fich);
    fich.close();

    std::size_t resx=800, resy=600;
    if (nargs>3)
    {
        resx = std::stoull(argv[2]);
        resy = std::stoull(argv[3]);
    }

    auto vortices = std::get<0>(config);
    auto isMobile = std::get<1>(config);
    auto grid     = std::get<2>(config);
    auto cloud    = std::get<3>(config);


    double dt = 0.1;

    //Orders to be exchange between the processes (Continuous animation, Step animation, dt)
    //orders[0] -> Continuous animation{0:false - 1:true}
    //orders[1] -> Step animation{0:false - 1:true}
    //orders[2] -> dt
    double orders[3];
    grid.updateVelocityField(vortices);

    //The rank 0 will be in charge of the display
    if (rank==0){
        std::cout << "######## Vortex simulation ########" << std::endl << std::endl;
        std::cout << "Press P for play animation " << std::endl;
        std::cout << "Press S to stop animation" << std::endl;
        std::cout << "Press right cursor to advance step by step in time" << std::endl;
        std::cout << "Press down cursor to halve the time step" << std::endl;
        std::cout << "Press up cursor to double the time step" << std::endl;

        Graphisme::Screen myScreen( {resx,resy}, {grid.getLeftBottomVertex(), grid.getRightTopVertex()} );

        while (myScreen.isOpen())
        {
            
            auto start = std::chrono::system_clock::now();
            // on inspecte tous les évènements de la fenêtre qui ont été émis depuis la précédente itération
            sf::Event event;

           
            while (myScreen.pollEvent(event))
            {
                
                // évènement "fermeture demandée" : on ferme la fenêtre
                if (event.type == sf::Event::Closed)
                {
                    myScreen.close();

                    //For stoping MPI when we close the window
                    MPI_Abort(MPI_COMM_WORLD, 0);
                    MPI_Finalize();
                }
                if (event.type == sf::Event::Resized)
                {
                    // on met à jour la vue, avec la nouvelle taille de la fenêtre
                    myScreen.resize(event);
                }

                //When we press P: 
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
                {    
                    
                    //Update of the orders when pressed P
                    orders[0] = 1;
                    orders[1] = 0;
                    orders[2] = dt;
                }

                //When press S
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                {
                    
                    //Update of the orders when pressed S
                    orders[0] = 0;
                    orders[1] = 0;
                    orders[2] = dt;
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) dt *= 2;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) dt /= 2;
                
                //When press Right arrow
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
                {   
                    //Update of the orders when pressed Right arrow
                    orders[0] = 0;
                    orders[1] = 1;
                    orders[2] = dt;
                }


            }

            //Whenever either Continuous or step animation are true:1
            if((orders[0] == 1 )| (orders[1] == 1)){
                
                //Sends the array with the orders
                MPI_Send(&orders, 3, MPI_DOUBLE, 1, 101, MPI_COMM_WORLD);

                //Wait for the values of cloud, vortices, grid and the updated orders from rank 1
                MPI_Recv(cloud.data(), cloud.numberOfPoints()*2, MPI_DOUBLE, 1, 102, MPI_COMM_WORLD, &status);
                MPI_Recv(vortices.data(), vortices.numberOfVortices()*3, MPI_DOUBLE, 1, 103, MPI_COMM_WORLD, &status);
                MPI_Recv(grid.data(), grid.cellGeometry().first* grid.cellGeometry().second*2, MPI_DOUBLE, 1, 104, MPI_COMM_WORLD, &status);
                MPI_Recv(&orders, 3, MPI_DOUBLE, 1, 105, MPI_COMM_WORLD, &status);
            }

            myScreen.clear(sf::Color::Black);
            std::string strDt = std::string("Time step : ") + std::to_string(dt);
            myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second-96)});
            myScreen.displayVelocityField(grid, vortices);
            myScreen.displayParticles(grid, vortices, cloud);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::string str_fps = std::string("FPS : ") + std::to_string(1./diff.count());
            myScreen.drawText(str_fps, Geometry::Point<double>{300, double(myScreen.getGeometry().second-96)});
            myScreen.display();

             
        }

    }

    //The rank 1 will be in charge of the calculation
    if (rank==1){
        while(true){

            //It takes the orders from rank 0
            MPI_Recv(&orders, 3, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, &status);
            
            ////Whenever either Continuous or step animation are true:1
            if ((orders[0] == 1) | (orders[1] == 1))
            {
                if (isMobile)
                {
                    cloud = Numeric::solve_RK4_movable_vortices(orders[2], grid, vortices, cloud);
                
                }
                else
                {
                    cloud = Numeric::solve_RK4_fixed_vortices(orders[2], grid, cloud);
                }

                //Update orders when for the step animation information
                orders[1] = 0;

                //Sends the cloud, vortices, grid and update orders
                MPI_Send(cloud.data(), cloud.numberOfPoints()*2, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD);
                MPI_Send(vortices.data(), vortices.numberOfVortices()*3, MPI_DOUBLE, 0, 103, MPI_COMM_WORLD);
                MPI_Send(grid.data(), grid.cellGeometry().first * grid.cellGeometry().second * 2, MPI_DOUBLE, 0, 104, MPI_COMM_WORLD);
                MPI_Send(&orders, 3, MPI_DOUBLE, 0, 105, MPI_COMM_WORLD);
                
            }

        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
 }
