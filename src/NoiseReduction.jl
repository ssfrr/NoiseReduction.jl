module NoiseReduction

using SFRDSP
using SampledSignals
using DSP
using TensorCast
using Statistics
using TSVD
using LinearAlgebra

export spec_sub, wiener, lowrank

function _spec_sub(X, fr, smoothing, α, β)
    # implements spectral subtraction from (Martin 1994)
    X2 = abs2.(X)
    npow = noise_psd(X, smoothing, fr)
    # α is an "oversubtract" parameter to compensate for the underestimated
    # noise power and is typically in the range of 2-6
    # β is a "noise floor" parameter that will limit how low this will push result,
    # and is usually in the range  0 < β << 1
    # taking the minimum of X2 and the noise floor here is a slight tweak on the
    # original in (berouti 1979), because if the signal happens to be below the noise
    # noise floor the given equation would end up with a bin with greater energy than
    # the mixture. This is particularly problematic when you're using this to estimate
    # the wiener filter.
    X2_sub = @. max(min(X2, β*npow), X2 - α * npow)
    @. sqrt(X2_sub) * exp(im*angle(X))
end

"""
Perform spectral subtraction by first estimating the noise power spectral density, then
subtracting the noise energy from the mixture energy.

`audio`            - a SampleBuf
`nfft`             - The FFT size of the STFT
`hop`              - The hop size of the STFT
`noise_smoothing`  - The window size (in seconds) used to smooth each band before taking the
                     minimum
`oversub`          - Apply this multiplier to the noise estimate before subtracting. Typically
                     in the range of 1-6.
`specfloor`        - Use this (multiplied by the noise power) as a lower bound on each band
                     after subtraction. This is used to limit "musical noise". Typically in
                     the range of 0.005 to 0.1.
"""
function spec_sub(audio;
                  nfft=1024, hop=nfft÷2,
                  noise_smoothing=0.25, oversub=4, specfloor=0.01)
    fs = samplerate(audio)
    X = stft2(audio, nfft, hop; window=cosine)
    X_ss = _spec_sub(X, fs/hop, noise_smoothing, oversub, specfloor)
    # N_ss = X - X_ss
    x_ss = SampleBuf(istft2(X_ss, nfft, hop; window=cosine)[1:length(audio)], fs)
    # n_ss = SampleBuf(istft2(N_ss, nfft, hop; window=cosine)[1:length(audio)], fs)
    n_ss = audio-x_ss
    x_ss, n_ss
end

"""
Perform noise reduction by Wiener Filtering

`audio`            - a SampleBuf
`nfft`             - The FFT size of the STFT
`hop`              - The hop size of the STFT
`noise_smoothing`  - The window size (in seconds) used to smooth each band before taking the
                     minimum
`oversub`          - Apply this multiplier to the noise estimate before subtracting. Typically
                     in the range of 1-6.
`specfloor`        - Use this (multiplied by the noise power) as a lower bound on each band
                     after subtraction. This is used to limit "musical noise". Typically in
                     the range of 0.005 to 0.1.
`wf_smoothing`     - 2D Smoothing applied to the wiener filter before applying (in
                     samples). Set to `1` to disable smoothing.
"""

function wiener(audio; nfft=1024, hop=nfft÷2,
                noise_smoothing=0.25,
                oversub=4, specfloor=0.01, wf_smoothing=11)
    @assert isodd(wf_smoothing)
    fs = samplerate(audio)
    wf_L = wf_smoothing
    X = stft2(audio, nfft, hop; window=cosine)
    # note that we could do this more efficiently in a single step rather than
    # doing the spectral subtraction and then computing the wiener filter
    # coefficient. Not sure if we can incorporate the α and β params though
    X_ss = _spec_sub(X, fs/hop, noise_smoothing, oversub, specfloor)

    # compute and smooth the wiener filter
    wf = @. abs2(X_ss) / abs2(X)
    win = gaussian((wf_L, wf_L), (0.15, 0.15))
    win ./= sum(win)
    wf_sm = conv(win, wf)[wf_L÷2+1:end-wf_L÷2, wf_L÷2+1:end-wf_L÷2]

    # apply the wiener filter
    X_wf = X .* wf_sm

    x_wf = SampleBuf(istft2(X_wf, nfft, hop; window=cosine)[1:length(audio)], fs)
    x_wf, audio-x_wf
end

"""
Performs a low-rank projection across time and channels in each frequency band. Assumes
the input audio has been pre-aligned.

Returns a `Vector` of de-noised signals, as well as a `Vector` of the noise channels,
which are each channel projected into the noise subspace.
"""
function lowrank(audio, rank=1; nfft=1024, hop=nfft÷2,
                 tol=1e-5, niters=20,
                 weight_smoothing=31,
                 wiener=true, wiener_alpha=2, l1_weight=false,
                 normalize=:none,
                 weight_pow=0.5,
                 noise_rank=length(audio)-rank)
    fs = first(unique(samplerate.(audio)))
    @assert(all(samplerate.(audio) .== fs))

    @cast Xs[k, n, c] := stft2(audio[c], nfft, hop; window=cosine, demod=true)[k, n]
    @cast Npsd[k, c] := noise_psd(Xs[:, :, c], 0.25, fs/hop)[k]
    K, N, C = size(Xs)

    Xs_wf = if wiener
        @reduce Xpsd[k, c] := mean(n) abs2(Xs[k, n, c])
        @cast wf[k, c] := max(0.0, Xpsd[k, c] - wiener_alpha*Npsd[k, c]) / Xpsd[k, c]
        @cast [k, n, c] := Xs[k, n, c] * wf[k,c]
    else
        Xs
    end

    # normalize each channel so the louder ones don't dominate the SVD.
    # also flip the dimensions here for better performance downstream
    Xs_norm = if normalize == :channel
        @reduce Xpow[c] := mean(k,n) max(1e-8, abs2(Xs_wf[k, n, c]))
        @cast [c, n, k] := Xs_wf[k, n, c] / Xpow[c]
    elseif normalize == :band
        @reduce Xpow[c, k] := mean(n) max(1e-8, abs2(Xs_wf[k, n, c]))
        @cast [c, n, k] := Xs_wf[k, n, c] / Xpow[c, k]
    elseif normalize == :none
        permutedims(Xs_wf, (3,2,1))
    else
        throw(ArgumentError("Unrecognized `normalize` value $normalize"))
    end

    # we smooth the window across frequency so we can accentuate bins that are
    # nearby signal bins above or below - this helps with some of the horizontal
    # banding
    sm_L=weight_smoothing
    win = reshape(gaussian(sm_L, 0.15), 1, :)
    win ./= sum(win)

    # weights for each time-frequency bin
    w = ones(N, K)
    w_last = copy(w)
    dw = Inf

    # some buffers so we don't need to allocate in the loop as much
    Xbuf = similar(Xs_norm, (C, N))
    Xp = similar(Xs_norm, (rank, N))

    # this doesn't necessarily converge monotonically, but hopefully the change
    # gets pretty small
    # if the weight power is 0 then we don't need to iterate, because we're not
    # refining the weights
    while weight_pow > 0 && dw > tol && niters > 0
        for k in 1:K
            # we're computing a low-rank approximation, and updating the weights based
            # on how much of each time point is in our signal subspace.
            @cast Xbuf[c, n] = Xs_norm[c, n, $k] * w[n, $k]
            U, S, V = tsvd(Xbuf, rank)
            @cast Xbuf[c, n] = Xs_norm[c, n, $k]
            mul!(Xp, U', Xbuf)
            # we're weighting by the normalized energy in the signal subspace here.
            # It was previously the square-root, or amplitude ratio. Not clear which
            # is better.
            w[:, k] .= vec(sum(abs2, Xp; dims=1) ./
                           sum(abs2, Xbuf; dims=1)) .^ weight_pow
        end
        # smooth the weights across frequency
        w = conv(win, w)[:, sm_L÷2+1:end-sm_L÷2]

        dw = norm(w .- w_last) / length(w)
        @info "weights changed by $dw"
        w_last .= w
        niters -= 1
    end

    Xs_proj = similar(Xs)
    Ns_proj = similar(Xs)
    # perform the final iteration and project back into the multichannel space
    for k in 1:K
        # X = @view Xs_norm[k, :, :]
        @cast Xbuf[c, n] = Xs_norm[c, n, $k] * w[n, $k]
        # U, S, V = tsvd(Xbuf, rank)
        D = svd(Xbuf)
        Ux = D.U[:, 1:rank]
        Un = D.U[:, end-noise_rank+1:end]

        # previously I had the commented-out projection, but now (Mar 9, 2020) I'm
        # thinking that actually I want to project the signal, rather than just
        # reconstructing from the SVD (which includes the weights.)
        # Xk = D.U[:, 1:rank] * D.S[1:rank] * D.Vt[1:rank, :]
        # nr = rank+C-noise_rank
        # Nk = D.U[:, nr:end] * (D.S[nr:end] .* D.Vt[nr:end, :])

        @cast Xbuf[c, n] = Xs_norm[c, n, $k]
        Xk = Ux * Ux' * Xbuf
        Nk = Un * Un' * Xbuf
        @cast Xs_proj[$k, n, c] = Xk[c, n] * w[n, $k]
        @cast Ns_proj[$k, n, c] = Nk[c, n]
    end

    if normalize == :channel
        @cast Xs_proj[k, n, c] = Xs_proj[k, n, c] * Xpow[c]
        @cast Ns_proj[k, n, c] = Ns_proj[k, n, c] * Xpow[c]
    elseif normalize == :band
        @cast Xs_proj[k, n, c] = Xs_proj[k, n, c] * Xpow[c, k]
        @cast Ns_proj[k, n, c] = Ns_proj[k, n, c] * Xpow[c, k]
    end

    @cast target[c] := SampleBuf(istft2(Xs_proj[:, :, c], nfft, hop;
                                        window=cosine, demod=true),
                                 fs)

    # we also add back the noise removed by the wiener filter
    # TODO: this isn't going to be scaled properly if there was normalization
    @cast Ns_proj[k, n, c] = Ns_proj[k, n, c] + (Xs[k, n, c] - Xs_wf[k, n, c])
    @cast noise[c] := SampleBuf(istft2(Ns_proj[:, :, c], nfft, hop;
                                       window=cosine, demod=true),
                                fs)

    target, noise
end

end # module
