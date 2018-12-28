import time
import torch
import torch.nn as nn

def train(encoder, decoder, train_iter, valid_iter, args):

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    steps = 0
    last_step = 0

    # log_file = open('log.txt', 'w')
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            post, response = batch.post, batch.response
            encoder_optimizer.zero_grad()
            criterion = nn.NLLLoss()


# def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0  # Reset every print_every
#     plot_loss_total = 0  # Reset every plot_every
#
#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#     training_pairs = [tensorsFromPair(random.choice(pairs))
#                       for i in range(n_iters)]
#     criterion = nn.NLLLoss()
#
#     for iter in range(1, n_iters + 1):
#         training_pair = training_pairs[iter - 1]
#         input_tensor = training_pair[0]
#         target_tensor = training_pair[1]
#
#         loss = train(input_tensor, target_tensor, encoder,
#                      decoder, encoder_optimizer, decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss
#
#         if iter % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                          iter, iter / n_iters * 100, print_loss_avg))
#
#         if iter % plot_every == 0:
#             plot_loss_avg = plot_loss_total / plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total = 0