#pragma once
#include "core/math_utils.hpp"
#include "foundations/volatility.hpp"
#include "risk/var_cvar.hpp"
#include "strategies/systematic_engine.hpp"
#include "derivatives/black_scholes_heston.hpp"
#include "macro_fixed_income/term_structure_carbon.hpp"
#include "microstructure_execution/hft_engine.hpp"
#include "ai_alternative_data/ml_gnn_swarm.hpp"
#include "defi_prediction_markets/amm_perp_arb.hpp"
#include "equities_etf/institutional_etf_engine.hpp"
#include "rigor_validation/deflated_sharpe_pipeline.hpp"
#include "orchestrator/nexus_master_engine.hpp"
#include "hft_ultralow/fixed_point.hpp"
#include "hft_ultralow/spsc_ring_buffer.hpp"
#include "hft_ultralow/flat_limit_order_book.hpp"
#include "hft_ultralow/fast_market_maker.hpp"
#include "hft_ultralow/itch_parser.hpp"


namespace nq = nexus_quant;
