"""
FNO for 2D problem such as the Navier-Stokes equations
(section 5.3 in https://arxiv.org/pdf/2010.08895.pdf),
uses a recurrent structure to propagates in time
"""

# libraries
from FNO2d import FNO2d
from Adam import Adam
from torch.autograd import Variable

# third-party libraries
import torch as th
from utilities3 import *
from timeit import default_timer
from fig import Plot_Loss, Field, Plot_1D, Plot_45

# choose device
is_gpu  = True
if is_gpu and th.cuda.is_available():
    device = "cuda"
    print("Working on GPU", flush=True)
else:
    device = "cpu"
    print("Working on CPU", flush=True)

# path to data
train_dir = 'data/data.mat'
valid_dir = 'data/data.mat'
test_dir  = 'data/data_pred.mat'
work_dir  = 'work/'

# parameters
ntrain     = 540        # number of training fields
nvalid     = 20         # number of valid fields
epochs     = 300        # number of training epochs
learn_rate = 5.e-3      # learning rate
sch_gamma  = 0.85       # decay factor
T_in       = 10         # number of timesteps in input
T_out      = 10         # number of timesteps in output
step       = 1          # steps between time steps

modes      = 12          # number of Fourier model to apply: 12 before
width      = 20         # number of input / output channels
batch_size = 50         # batch size
weight_decay = 0.0
beta = 1.e-4

# set the regime
is_train = True
is_load  = True       # load before (to continue training)
is_test  = True

# train
if is_train:

    # load training data
    print("Reading training data", flush=True)
    reader = MatReader(train_dir)
    train_inp = reader.read_field('fi')[:ntrain, :, :, :T_in]
    train_tar = reader.read_field('fi')[:ntrain, :, :,  T_in:T_out+T_in]
    train_u_i = reader.read_field('u')[:ntrain, :, :, :T_in]
    train_u_t = reader.read_field('u')[:ntrain, :, :,  T_in:T_out+T_in]
    mesh_x    = reader.read_field('x')
    mesh_y    = reader.read_field('y')
    print("    training inputs shape  :",     train_inp.cpu().numpy().shape, flush=True)
    print("    training targets shape :",     train_tar.cpu().numpy().shape, flush=True)
    print("    training vel inputs shape :",  train_u_i.cpu().numpy().shape, flush=True)
    print("    training vel targets shape :", train_u_t.cpu().numpy().shape, flush=True)
    print("    mesh elements in x shape :",   mesh_x.cpu().numpy().shape,    flush=True)
    print("    mesh elements in y shape :",   mesh_y.cpu().numpy().shape,    flush=True)
    train_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(train_inp, train_tar, train_u_i, train_u_t), batch_size=batch_size, shuffle=True)

    # load validing data
    print("Reading validation data")
    reader = MatReader(valid_dir)
    valid_inp = reader.read_field('fi')[-nvalid:, :, :, :T_in]
    valid_tar = reader.read_field('fi')[-nvalid:, :, :,  T_in:T_out+T_in]
    valid_u_i = reader.read_field('u')[-nvalid:, :, :, :T_in]
    valid_u_t = reader.read_field('u')[-nvalid:, :, :,  T_in:T_out+T_in]    
    print("    validation inputs shape  :",     valid_inp.cpu().numpy().shape, flush=True)
    print("    validation targets shape :",     valid_tar.cpu().numpy().shape, flush=True)
    print("    validation vel inputs shape :",  valid_u_i.cpu().numpy().shape, flush=True)
    print("    validation vel targets shape :", valid_u_t.cpu().numpy().shape, flush=True)    
    valid_loader  = th.utils.data.DataLoader(th.utils.data.TensorDataset(valid_inp, valid_tar, valid_u_i, valid_u_t),   batch_size=batch_size, shuffle=False)    

    # model
    model = FNO2d(modes, modes, width, mesh_x, mesh_y).cuda()

    # optimizer
    optimizer = Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    # learning rate decay scheduler
    #scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=sch_gamma, patience=10, verbose=True, 
                                                        threshold=1.e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1.e-7)

    # create or load model
    if is_load:
        print("Loading model", flush=True)
        #model = th.load(work_dir + 'model.pth')
        Load_NN(work_dir + 'model.pth', device, model, optimizer, scheduler)
            
    print(model, flush=True)

    # loss function
    myloss = MSE_loss() #LpLoss(size_average=False)    

    print("Begin training", flush=True)

    # mesh for gradient
    x_all = th.unique(mesh_x.round(decimals=6)).to(device)
    y_all = th.unique(mesh_y.round(decimals=6)).to(device)
    coords = (x_all, y_all)

    # loop over epochs
    best_loss  = 1.e6
    lr_hist       = np.zeros((epochs))
    train_l2_step = np.zeros((epochs))  # per one time step
    train_l2_full = np.zeros((epochs))  # per all time steps
    valid_l2_step = np.zeros((epochs))   # per one time step
    valid_l2_full = np.zeros((epochs))   # per all time steps
    for epoch in range(epochs):

        # measure execution time
        t1 = default_timer()

        # move to the training state
        model.train()

        # loop over batch
        for xx, yy, ui, ut in train_loader:

            # move to device
            xx = xx.to(device)  # input level set
            yy = yy.to(device)  # target level set
            ui = ui.to(device)  # input velocity
            ut = ut.to(device)  # target velocity

            # loop over time steps
            loss = 0.0
            for t in range(0, T_out, step):

                u = th.cat((ui[..., step-1+t:], ut[..., :step-1+t]), dim=-1)
                u = u.to(device)
                y = yy[..., t:t + step]
                x = th.cat((xx, u), dim=-1)     # [batch,nx,ny,10+10]
                im = model(x)                   # [batch,nx,ny,1]

                # gradient
                dfidx = Variable(th.zeros((im.shape[0], im.shape[1], im.shape[2])), requires_grad=True).to(device)
                dfidy = Variable(th.zeros((im.shape[0], im.shape[1], im.shape[2])), requires_grad=True).to(device)
                for ig in range(im.shape[0]):
                    dfidx[ig,:,:] = th.gradient(im[ig,:,:,0], spacing=coords)[0]
                    dfidy[ig,:,:] = th.gradient(im[ig,:,:,0], spacing=coords)[1]

                magnfi = Variable(th.sqrt(dfidx**2 + dfidy**2), requires_grad=True).to(device)
                #x_c  = np.meshgrid(mesh_y, mesh_x)[1]
                #y_c  = np.meshgrid(mesh_y, mesh_x)[0]
                #Field(x_c.flatten(), y_c.flatten(), magnfi.numpy().flatten(), magnfi.numpy().flatten(), str(10000))

                #loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                loss += myloss(im, y, magnfi, beta)
                if t == 0:
                    pred = im
                else:
                    pred = th.cat((pred, im), -1)
                xx = th.cat((xx[..., step:], im), dim=-1)

            # save current loss
            train_l2_step[epoch] += loss.item()/ntrain/(T_out/step)
            train_l2_full[epoch] += myloss(pred, yy, magnfi, beta).item()/ntrain

            # train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # loop over batch
        with th.no_grad():
            for xx, yy, ui, ut in valid_loader:

                # move to device
                xx = xx.to(device)
                yy = yy.to(device)

                # loop over time steps
                valid_loss = 0
                for t in range(0, T_out, step):

                    u = th.cat((ui[..., step-1+t:], ut[..., :step-1+t]), dim=-1)
                    u = u.to(device)
                    y = yy[..., t:t + step]
                    x = th.cat((xx, u), dim=-1)     # [batch,nx,ny,10+10]
                    im = model(x)                   # [batch,nx,ny,1]

                    # gradient
                    dfidx = Variable(th.zeros((im.shape[0], im.shape[1], im.shape[2])), requires_grad=True).to(device)
                    dfidy = Variable(th.zeros((im.shape[0], im.shape[1], im.shape[2])), requires_grad=True).to(device)
                    for ig in range(im.shape[0]):
                        dfidx[ig,:,:] = th.gradient(im[ig,:,:,0], spacing=coords)[0]
                        dfidy[ig,:,:] = th.gradient(im[ig,:,:,0], spacing=coords)[1]
                    magnfi = Variable(th.sqrt(dfidx**2 + dfidy**2), requires_grad=True).to(device)

                    valid_loss += myloss(im, y, magnfi, beta)
                    if t == 0:
                        pred = im
                    else:
                        pred = th.cat((pred, im), -1)
                    xx = th.cat((xx[..., step:], im), dim=-1)

                # save current loss
                valid_l2_step[epoch] += valid_loss.item()/nvalid/(T_out/step)
                valid_l2_full[epoch] += myloss(pred, yy, magnfi, beta).item()/nvalid
        
            # save best NN
            if best_loss > valid_l2_full[epoch]:

                best_loss = valid_l2_full[epoch]
                best_epoch = epoch

                # save model
                print("Saving model. Loss:", round(best_loss,6), flush=True)
                Save_NN(model, optimizer, scheduler, work_dir + 'model.pth', work_dir + 'model.pt',  x.shape, device, is_ts=False)

        # update learing rate
        lr_hist[epoch] = optimizer.param_groups[0]['lr']
        scheduler.step(best_loss)

        # measure execution time
        t2 = default_timer()

        # print info
        if (epoch % 10 == 0):
            print('--------------------------------------------------------------------', flush=True)
            print('epoch -', 'time/epoch -', 'loss/step -', 'loss -', 'valid loss/step -', 'valid loss', flush=True)
            print('--------------------------------------------------------------------', flush=True)
        print("{0:3d}     {1:4.2f}        {2:8.6f}     {3:8.6f}     {4:8.6f}     {5:8.6f}".format(epoch, 
              t2-t1, 
              train_l2_step[epoch],
              train_l2_full[epoch],
              valid_l2_step[epoch],
              valid_l2_full[epoch]), flush=True)

    print("Training is completed. Best epoch: ", best_epoch, flush=True)
    print("                       Best valid loss: ", round(best_loss,4), flush=True)
    print("                       Train loss there: ", round(train_l2_full[epoch],4), flush=True)

    print("Making figures")
    Plot_Loss(epochs, valid_l2_step, train_l2_step, lr_hist, "step")
    Plot_Loss(epochs, valid_l2_full, train_l2_full, lr_hist, "full")
        
if is_test:

    # load testing data
    print("Reading test data")

    # load training data
    reader = MatReader(test_dir)
    test_inp = reader.read_field('fi')
    test_tar = reader.read_field('fi')
    test_u_i = reader.read_field('u')
    test_u_t = reader.read_field('u')
    mesh_x   = reader.read_field('x')
    mesh_y   = reader.read_field('y')
    print("    test inputs shape  :",     test_inp.cpu().numpy().shape, flush=True)
    print("    test targets shape :",     test_tar.cpu().numpy().shape, flush=True)
    print("    test vel inputs shape :",  test_u_i.cpu().numpy().shape, flush=True)
    print("    test vel targets shape :", test_u_t.cpu().numpy().shape, flush=True)
    print("    mesh elements in x shape :",     mesh_x.cpu().numpy().shape,   flush=True)
    print("    mesh elements in y shape :",     mesh_y.cpu().numpy().shape,   flush=True)
    test_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(test_inp, test_tar, test_u_i, test_u_t), batch_size=batch_size, shuffle=False)

    print("Loading model", flush=True)
    model = FNO2d(modes, modes, width, mesh_x, mesh_y).cuda()
    optimizer = Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    #scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=sch_gamma, patience=10, verbose=True, 
                                                        threshold=1.e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1.e-7)
    Load_NN(work_dir + 'model.pth', device, model, optimizer, scheduler)

    # update mesh if changed
    model.mesh_x = mesh_x
    model.mesh_y = mesh_y
    print(model, flush=True)

    print("Begin test", flush=True)
    
    # loop over batch
    with th.no_grad():

        for xx, yy, ui, ut in test_loader:

            # move to device
            xx = xx.to(device)
            yy = yy.to(device)
            ui = ui.to(device)
            ut = ut.to(device)

            # take first 10 steps and proceed in time
            xx_10 = xx[..., 0:T_in]
            ui_10 = ui[..., 0:T_in]

            # save 10 steps in prediction
            pred = xx_10

            # loop over time steps
            for t in range(0, yy.shape[-1]-T_in, 1):

                # propogate
                x = th.cat((xx_10, ui_10), dim=-1)  # [1,nx,ny,10+10]
                im = model(x)                       # [1,nx,ny,1]

                # save prediction
                pred  = th.cat((pred, im), -1)

                # proceed in time
                xx_10 = th.cat((xx_10[..., step:], im), dim=-1)

            # plot field
            for j in range(T_in, yy.shape[-1], 10):
                x_c  = np.meshgrid(mesh_y, mesh_x)[1]
                y_c  = np.meshgrid(mesh_y, mesh_x)[0]
                pr   = pred.cpu().numpy()[0,:,:,j]
                tr   = yy.cpu().numpy()[0,:,:,j]
                Field(x_c.flatten(), y_c.flatten(), pr.flatten(), tr.flatten(), str(j))
                Plot_45(pr.flatten(), tr.flatten(), str(j) + '_45')
                ind_c = x_c.shape[1] // 2
                Plot_1D(x_c[:,ind_c], pr[:,ind_c], tr[:,ind_c], str(j))

    # saving test fields
    fname = 'work/results_' + str(ntrain) + '_' + str(nvalid)
    np.savez(fname, x_c=x_c, y_c=y_c, pr=pr, tr=tr)

    # load files
    #data = np.load(fname + '.npz')
    #x_c_red = data['x_c']

    print("test is completed")